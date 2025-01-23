from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import os
# from pdf2image import convert_from_bytes
import fitz  # PyMuPDF
import openai
from openai import AzureOpenAI
import base64

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

# Azure OpenAI を利用するとき
# openai.api_type = "azure"
url = os.getenv('OPENAI_API_URL')   # エンドポイント
key = os.getenv('OPENAI_API_KEY')  # 環境変数からAPIキーを取得
version = "2024-08-01-preview"

storage_connection = os.getenv('BLOB_CONNECTION')
# アップロード先のコンテナー名
BLOB_CONTAINER_NAME = "4o-pdf2image-converted" 

@csrf_exempt  # 開発用のCSRF無効化（本番では必ず有効化してください）
def index(request):
    """"
    txt ファイルからプロンプトを読みだして、index.html を表示する
    """

    try:
        with open('prompt.txt', 'r', encoding='utf-8') as file:
            content = file.read()
    except FileExistsError:
        content = "ファイルが見つかりませんでした。"
    except Exception as e:
        content = f"エラーが発生しました: {e}"

    return render(request, 'myapp/index.html', {'file_content': content})

def pdf_to_jpegs(pdf_bytes):
    images = []
    # PDFをメモリ上でオープン（bytestream, filetype="pdf"）
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            # ピクセルマップ（Pix）を取得
            pix = page.get_pixmap()
            # JPEG形式のバイト列を作成
            img_bytes = pix.pil_tobytes(format="JPEG")
            images.append(img_bytes)
    return images

def call_4o_model_with_image_url(jpeg_path_list, base_name, systemprompt, temperature, max_tokens, top_p):
    print("JPEG_URL:", jpeg_path_list[0])
    client = AzureOpenAI(
        api_key=key,
        api_version=version,
        azure_endpoint=url
    )
    print(client._base_url)

    content_list = []
    content_list.append({
                "type": "text", 
                "text": systemprompt}
                )

    # Blob Storage から画像を読み込む
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection)
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

    for i, _ in enumerate(jpeg_path_list):
        blob_client = container_client.get_blob_client(f"{base_name}/page{i+1}.jpg")
        
        # Blob ファイルをダウンロード
        downloaded_blob = blob_client.download_blob()
        image_data = downloaded_blob.readall()  # バイナリ形式で読み込む

        # Base64 にエンコード
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })

    # メッセージリスト
    messages=[
        {
        "role": "system",
        "content": "あなたは日本人です。"
        },
        {
        "role": "user",
        "content": content_list
        }
    ]

    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

# Blob Storage の接続設定を行う
def get_blob_service_client():
    connection_string = storage_connection
    if not connection_string:
        raise ValueError("Azure Storageの接続文字列が環境変数 'AZURE_STORAGE_CONNECTION_STRING' に設定されていません。")
    return BlobServiceClient.from_connection_string(connection_string)

def upload_jpegs_to_blob(jpegs, base_name):
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
    
    # アップロードされたJPEGのURLリスト
    image_urls = []
    
    for i, img_bytes in enumerate(jpegs):
        filename = f"page{i+1}.jpg"
        blob_name = f"{base_name}/{filename}"  # 仮想ディレクトリとして <base_name>/page1.jpg の形式にする
        print("Blob_name", blob_name)
        blob_client = container_client.get_blob_client(blob_name)
        
        try:
            # Blobが既に存在する場合のハンドリング（オプション）
            blob_client.upload_blob(img_bytes, overwrite=True)
            # BlobのURLを生成
            blob_url = blob_client.url
            image_urls.append(blob_url)
        except ResourceExistsError:
            print(f"Blob {blob_name} は既に存在します。上書きします。")
            blob_client.upload_blob(img_bytes, overwrite=True)
            blob_url = blob_client.url
            image_urls.append(blob_url)
        except Exception as e:
            print(f"Blob {blob_name} のアップロード中にエラーが発生しました: {e}")
    
    return image_urls

@csrf_exempt  # 開発用のCSRF無効化（本番では必ず有効化してください）
def upload(request):
    """
    1. どんなファイルでもアップロードを受け取る
    2. PDFファイルなら pdf2image でJPEGに変換
    3. PDFのファイル名を基にディレクトリを作成し、そこにJPEGを保存
    4. 成功/失敗のメッセージを返す
    """
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('files')
        if not uploaded_files:
            return HttpResponse("アップロードされたファイルがありません。")
        
        # プロンプトとパラメータを取得
        prompt_text = request.POST.get('promptText', '')

        try:
            temperature = float(request.POST.get('temperature', 0.5))
        except ValueError:
            temperature = 0.5

        try:
            max_tokens = int(request.POST.get('maxTokens', 300))
        except ValueError:
            max_tokens = 300

        try:
            top_p = float(request.POST.get('topP', 0.9))
        except ValueError:
            top_p = 0.9

        # 処理結果のリスト
        righttop_messages = []
        rightbottom_messages = []
        
        for file_obj in uploaded_files:
            if file_obj.name.lower().endswith('.pdf'):
                try:
                    pdf_bytes = file_obj.read()
                    righttop_messages.append(f"読み込み成功")
                    
                    # PDFをJPEGに変換
                    jpegs = pdf_to_jpegs(pdf_bytes) #リスト型でページごとに保持
                    righttop_messages.append(f"JPEG変換成功")
                    
                    # PDF名（拡張子なし）を取得
                    base_name = os.path.splitext(file_obj.name)[0]

                    print("#####", len(jpegs), base_name)

                    # JPEGをBlob Storageにアップロード
                    try:
                        image_urls = upload_jpegs_to_blob(jpegs, base_name)
                        righttop_messages.append(f"{file_obj.name}: Blob Storageへのアップロード成功")
                    except Exception as e:
                        righttop_messages.append(f"{file_obj.name} のBlob Storageへのアップロードに失敗しました。エラー: {str(e)}")
                    
                    # gpt のレスポンス
                    try:
                        # system_prompt = "提供された複数枚の画像は1つのスライドの各ページとなっています。スライドの内容について説明してください。"
                        response = call_4o_model_with_image_url(image_urls, base_name, prompt_text, temperature, max_tokens, top_p)
                        # rightbottom_messages.append(f"プロンプト: {prompt_text}")
                        rightbottom_messages.append(f"temperature: {temperature}    max_token: {max_tokens}    top_p: {top_p}")
                        rightbottom_messages.append(f"<h4>モデル応答</h4> {response.choices[0].message.content}")
                    except Exception as e:
                        rightbottom_messages.append(f"{file_obj.name} のGPT応答の取得に失敗しました。エラー: {str(e)}")
                    
                    # # 成功メッセージを追加
                    # response_messages.append(f"{file_obj.name} の変換に成功しました。")                
                except Exception as e:
                    # 失敗メッセージを追加
                    righttop_messages.append(f"{file_obj.name} の処理中にエラーが発生しました。エラー: {str(e)}")
            else:
                # PDF以外のファイル
                righttop_messages.append(f"{file_obj.name} はPDFファイルではありません。")
        
        # 結果を単純に表示（ファイル名の列挙はなし）
        top_msg = "<br><br>".join(righttop_messages)
        bottom_msg = "<br><br>".join(rightbottom_messages)

        msg = {
            "topText": top_msg,
            "bottomText": bottom_msg
            }
        return JsonResponse(msg)
    
    else:
        return JsonResponse("POSTメソッドでアップロードしてください")