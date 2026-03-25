from synth_agent.flow.plan_flow import create_plan_flow

# 创建 PlanFlow
flow = create_plan_flow(
    model="Pro/MiniMaxAI/MiniMax-M2.5",
    api_key="sk-qqvwbybttiknbxvsxqujxrhmvlvdsaeqcmqtxetdiguimzym",
    base_url="https://api.siliconflow.cn/v1",
)

# 执行任务
result = flow.run("帮我写一份ai相关的研究，并保存到项目的result目录下")

# 查看依赖图
print(flow.visualize_plan())