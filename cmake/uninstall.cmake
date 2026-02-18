if(NOT EXISTS "${CMAKE_BINARY_DIR}/install_manifest.txt")
  message(FATAL_ERROR "Cannot find install_manifest.txt. Run 'make install' first.")
endif()

file(READ "${CMAKE_BINARY_DIR}/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" files "${files}")

foreach(file ${files})
  if(EXISTS "${file}" OR IS_SYMLINK "${file}")
    message(STATUS "Removing ${file}")
    file(REMOVE "${file}")
  endif()
endforeach()
