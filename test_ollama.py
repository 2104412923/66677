"""
Ollama API连通性测试脚本
功能：验证Ollama服务是否正常运行，模型是否可用
"""
import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "deepseek-r1:7b"

def test_ollama_connection():
    """测试Ollama服务是否在运行"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"[OK] Ollama服务运行正常")
            print(f"[OK] 已安装模型列表:")
            for model in models:
                print(f"      - {model['name']}")
            return True
        else:
            print(f"[ERROR] Ollama服务返回异常状态码: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[ERROR] 无法连接到Ollama服务，请确保Ollama已启动")
        print("        Windows: 在开始菜单中启动Ollama")
        print("        Linux/Mac: 运行 'ollama serve' 命令")
        return False
    except Exception as e:
        print(f"[ERROR] 连接测试失败: {e}")
        return False

def test_model_inference():
    """测试模型能否正常生成回答"""
    try:
        prompt = "请用一句话介绍自然语言处理（NLP）是什么。"
        print(f"\n[TEST] 测试模型推理能力")
        print(f"[PROMPT] {prompt}")
        print(f"[RESPONSE] ", end="", flush=True)

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "max_tokens": 200}
        }
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120
        )
        if response.status_code == 200:
            result = response.json().get("response", "")
            print(f"{result}")
            print(f"\n[OK] 模型推理测试通过")
            return True
        else:
            print(f"[ERROR] 模型推理失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] 模型推理测试异常: {e}")
        return False

def test_embeddings():
    """测试嵌入模型是否可用"""
    try:
        test_texts = ["自然语言处理技术", "深度学习"]
        payload = {
            "model": "nomic-embed-text",
            "prompt": test_texts[0]
        }
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            embedding = response.json().get("embedding", [])
            print(f"\n[OK] 嵌入模型测试通过，向量维度: {len(embedding)}")
            return True
        else:
            print(f"\n[WARN] 嵌入模型测试失败: {response.status_code}")
            print("        请确保已下载嵌入模型: ollama pull nomic-embed-text")
            return False
    except Exception as e:
        print(f"[WARN] 嵌入模型测试异常: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Ollama API 连通性测试脚本")
    print("=" * 60)

    conn_ok = test_ollama_connection()
    if conn_ok:
        test_model_inference()
        test_embeddings()

    print("\n" + "=" * 60)
    if conn_ok:
        print("结论: Ollama服务运行正常，可以开始构建RAG系统")
    else:
        print("结论: 请先确保Ollama服务已启动")
    print("=" * 60)