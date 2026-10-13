# Senior Shared - Time-Sensitive Fitness Service Platform

An AI multi-agent prototype for time-sensitive sharing of community fitness resources, designed with older adults in mind. Built with LangGraph, FastAPI, and a locally hosted DeepSeek model through Ollama, it demonstrates safe resource discovery, equipment status checks, reservation confirmation, companion matching, and human handoff for high-risk situations. Facility and booking data are simulated; the project is a runnable prototype and is not connected to real gym hardware or production services.

这是一个面向社区老年用户错时共享运动场景的 AI Multi-Agent 产品原型，使用 LangGraph、FastAPI 和通过 Ollama 本地运行的 DeepSeek 模型，演示共享运动资源推荐、设备状态查询、预约确认、同伴匹配及高风险请求人工接管。场馆、设备和预约数据均为演示模拟数据；项目可本地运行，但尚未接入真实场馆硬件或生产服务。

这是一个面向社区老年用户共享运动场景的 AI Agent 产品原型。它使用 LangGraph 编排路由、任务规划、上下文、风险、领域执行、冲突解决、人工接管和回复 Agent，底层模型通过本机 Ollama 运行，默认使用 `deepseek-r1:8b`；可通过 `OLLAMA_MODEL` 更换为已安装的 DeepSeek 模型标签。

本项目当前是**可运行的产品原型**，设备数据和预约数据默认保存在进程内，是为了展示产品流程和 Agent 编排，不代表已经接入真实场馆硬件。所有设备结果会标注“本地设备模拟器”，避免把演示数据误认为线上实时数据。

## 产品范围

当前支持：

- 搜索演示场馆、设施和营业时间。
- 查询设备状态，并展示数据来源、观测时间和新鲜度。
- 创建预约前补齐场馆、日期和时间，用户明确确认后才创建。
- 查询和取消当前用户的预约。
- 设备异常转运维工单，不直接执行开门、启动跑步机或调节速度等物理控制。
- 通过 SSE 展示路由、工具调用和最终回复。
- 识别高风险运动信号并停止常规推荐，转向家人、社区或专业人员支持。
- 推荐带距离、适老设施、时段、容量、同伴方式和风险说明的共享资源候选。
- 查询演示用户已确认的偏好与权限，匹配脱敏同伴候选。
- 预约后生成到场凭证、记录签到和提交反馈，形成“预约-到场-反馈”演示闭环。
- 路由后并行执行风险评估、确认档案/会话记忆读取和领域任务，再统一解决冲突。
- 高风险请求创建人工接管记录；普通请求不会被安全分支绕过。

## 架构

```text
Browser
  -> FastAPI /chat or /stream
  -> router_agent
  -> task_planner_agent
  -> context_agent + risk_agent + domain_agent (parallel)
  -> conflict_resolution_agent
  -> human_handoff_agent (when required)
  -> response_agent
  -> JSON or SSE
```

这是一个带并行 fan-out、共享状态和 fan-in 冲突解决的 Supervisor-Routed Agent Graph。高风险请求不会继续进入资源推荐或预约工具，而是进入人工接管节点。

## 运行前提

1. Python 3.10 或更高版本。
2. 已安装并启动 Ollama：`http://localhost:11434`。
3. 默认模型安装命令：`ollama pull deepseek-r1:8b`（也可在 `.env` 中设置 `OLLAMA_MODEL` 为其他已安装的 DeepSeek 标签）。

本地 Demo 会通过 `/auth/demo-token` 获取签名令牌。默认 Demo 登录密钥为 `local-demo-login`，仅用于本机演示；对外部署前必须替换 `AUTH_SECRET` 和 `DEMO_LOGIN_KEY`，并接入正式身份系统。

检查模型：

```powershell
ollama list
```

## 安装与启动

```powershell
cd "C:\Vscode\AI Agent（dong）\shared-fitness-multi-agent"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn multi_agent_service:app --host 127.0.0.1 --port 8010
```

打开：

- 工作台：<http://127.0.0.1:8010/>
- Swagger：<http://127.0.0.1:8010/docs>

## 配置

复制 `.env.example` 为 `.env`，可调整：

- `OLLAMA_BASE_URL`：Ollama 服务地址。
- `OLLAMA_MODEL`：本地模型名称。
- `OLLAMA_TIMEOUT`、`OLLAMA_RETRIES`：请求超时和重试次数。
- `DEVICE_DATA_MODE=simulator`：明确当前使用设备模拟器。
- `SESSION_TTL_SECONDS`、`MAX_SESSIONS`：本地会话过期和容量限制。
- `AUTH_SECRET`、`DEMO_LOGIN_KEY`：本地 Demo 令牌签名和引导密钥。
- `SESSION_DB_PATH`：SQLite 共享会话记忆文件路径。

## 预约安全流程

预约不会因为模型补全默认值而直接创建。流程是：

1. Agent 提取场馆、日期和时间。
2. 缺字段时追问用户。
3. 信息完整后展示预约摘要。
4. 用户回复“确认预约”。
5. 工具校验日期、营业时间、库存和幂等键后创建。
6. 返回预约编号，后续可查询或取消。

## API 示例

先获取本地 Demo 令牌；`/chat`、`/stream`、档案、权限和签到接口都要带 `Authorization`：

```powershell
$login = @{ "X-Demo-Login-Key" = "local-demo-login" }
$token = (Invoke-RestMethod http://127.0.0.1:8010/auth/demo-token?user_id=student-01 -Method Post -Headers $login).access_token
$auth = @{ Authorization = "Bearer $token" }
```

设备查询：

```powershell
Invoke-RestMethod http://127.0.0.1:8010/chat `
  -Method Post `
  -Headers $auth `
  -ContentType "application/json" `
  -Body '{"message":"跑步机现在有空位吗","user_id":"student-01"}'
```

预约第一步只会返回确认信息：

```powershell
$first = Invoke-RestMethod http://127.0.0.1:8010/chat `
  -Method Post `
  -Headers $auth `
  -ContentType "application/json" `
  -Body '{"message":"帮我预约明天下午3点西湖区健身中心","user_id":"student-01"}'

Invoke-RestMethod http://127.0.0.1:8010/chat `
  -Method Post `
  -Headers $auth `
  -ContentType "application/json" `
  -Body (@{message="确认预约"; user_id="student-01"; session_id=$first.session_id} | ConvertTo-Json)
```

## 测试

不启动模型即可运行结构和业务边界测试：

```powershell
.\.venv\Scripts\python.exe -B smoke_test.py
.\.venv\Scripts\python.exe -B evaluation_test.py
```

另开终端启动 Uvicorn 和 Ollama 后运行真实端到端测试：

```powershell
.\.venv\Scripts\python.exe -B api_test.py
```

真实测试覆盖健康检查、Agent 列表、并行协同事件、设备事实、预约确认、预约取消、会话越权、输入校验、共享资源推荐、安全分流和报修 SSE。

## 生产化差距

当前仍需接入真实硬件网关、正式身份提供商、家庭授权、预约库存服务、消息通知、监控指标、审计日志和 CI/CD。SQLite 适用于单机 Demo，多实例部署应替换为数据库和分布式锁。物理控制型工具必须增加设备前置状态、二次确认、幂等键、紧急停止和人工接管机制后，才能进入生产范围。具体的 PRD 增量映射见 `prd_gap_matrix.md`。

作品集说明见 `portfolio_case_study.md`，评测设计见 `evaluation.md`。