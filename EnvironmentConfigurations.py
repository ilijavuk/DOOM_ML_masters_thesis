EPISODES_NUM = 10
AGENT_MODEL_PATH_PREFIX = './agents/agent_for_'
TENSORBOARD_LOG_PATH_PREFIX = './logs/logs_for_'
CURRENT_CONFIGURATION_INDEX = 0
EVALUATION_FREQUENCY = 25000
MODEL_SAVING_FREQUENCY = 25000
EXPECTED_IMAGE_SHAPE = (240, 320, 3)

configurations = [{
                    'name': 'basic',
                    'scenarioConfigFilePath': 'VizDoom/scenarios/basic.cfg',
                    'actionNumber': 3,
                  }, {
                    'name': 'defend_the_center',
                    'scenarioConfigFilePath': 'VizDoom/scenarios/defend_the_center.cfg',
                    'actionNumber': 3,
                  }, {
                    'name': 'deadly_corridor',
                    'scenarioConfigFilePath': 'VizDoom/scenarios/deadly_corridor.cfg',
                    'actionNumber': 7,
                  }, {
                    'name': 'deathmatch',
                    'scenarioConfigFilePath': 'VizDoom/scenarios/deathmatch.cfg',
                    'actionNumber': 4, # 20
                  }, {
                    'name': 'defend_the_center_expanded',
                    'scenarioConfigFilePath': 'VizDoom/scenarios/defend_the_center_expanded.cfg',
                    'actionNumber': 4, # 20
                  }]
