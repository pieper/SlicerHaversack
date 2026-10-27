#-----------------------------------------------------------------------------
set(MODULE_NAME HaversackSegment)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/benchmark.py
  ${MODULE_NAME}Lib/catalog.py
  ${MODULE_NAME}Lib/environment.py
  ${MODULE_NAME}Lib/fetch_model.py
  ${MODULE_NAME}Lib/logtail.py
  ${MODULE_NAME}Lib/runner.py
  ${MODULE_NAME}Lib/stepprogress.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/extra_models.json
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )
