from mangum import Mangum
from ai_agents.api.main import app
from ai_agents.config.langsmith_bootstrap import ensure_langsmith_env

handler = Mangum(app)