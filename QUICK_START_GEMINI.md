# 快速开始：调用 Gemini

## 步骤 1：获取 API Key（如果还没有）

1. 访问 https://aistudio.google.com/
2. 登录 Google 账号
3. 点击 "Get API Key" 或 "Create API Key"
4. 复制 API key

## 步骤 2：设置环境变量

```bash
export GOOGLE_API_KEY="AIzaSyBPGiYIUgIy-mvAF2ifVNnileWksEtNE4k"
```

**永久设置**（推荐）：
```bash
echo 'export GOOGLE_API_KEY="你的API密钥"' >> ~/.bashrc
source ~/.bashrc
```

## 步骤 3：选择调用方式

### 方式 A：快速测试（推荐先试这个）

运行测试脚本验证 Gemini 是否工作：

```bash
cd /home/yilin/Agent-Omni/test
conda activate agent-omni
python test_gemini.py
```

### 方式 B：在 Agent-Omni 中使用

修改 `test/config.yaml`，将任意 agent 替换为 Gemini：

```yaml
model:
  master_agent: ../configs/model_configs/gemini-2.5-flash.yaml
  text_agent: ../configs/model_configs/gemini-2.5-flash.yaml
  # ... 其他 agent
```

然后运行：
```bash
cd /home/yilin/Agent-Omni/test
python agent_test.py
```

