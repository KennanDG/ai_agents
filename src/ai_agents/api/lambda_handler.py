from mangum import Mangum
from ai_agents.api.main import app

handler = Mangum(app)