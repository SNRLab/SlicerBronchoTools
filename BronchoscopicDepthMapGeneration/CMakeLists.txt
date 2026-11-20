#-----------------------------------------------------------------------------
set(MODULE_NAME BronchoscopicDepthMapGeneration)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/DepthEstimator.py
  ${MODULE_NAME}Lib/Networks.py
  )

# Sample data (a short porcine bronchoscopy video and its frames), included in the extension package so users
# can try the module. Re-run CMake after adding or removing sample data files.
file(GLOB_RECURSE MODULE_SAMPLE_DATA RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/Resources/SampleData/*)

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/Models.json
  ${MODULE_SAMPLE_DATA}
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
