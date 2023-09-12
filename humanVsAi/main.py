import threading
from humanVsAi.helpers.agent import startAgent
from humanVsAi.helpers.host import startHost

class Config:
  def __init__(self, configName, modelName):
    self.configName = configName
    self.modelName = modelName

configs = [
  Config("humanVsAi_deathmatch.cfg", "humanVsAi/model_deathmatch"),
  Config("humanVsAi_basic.cfg", "humanVsAi/model_basic")
]

currentConfig = configs[1]

threading.Thread(target=startHost, args=(currentConfig.configName,)).start()
startAgent(currentConfig.configName, currentConfig.modelName)