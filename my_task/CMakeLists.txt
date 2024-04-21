cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(my_task)

set(CMAKE_PREFIX_PATH "C:\\Libs\\libtorch-win-shared-with-deps-debug-2.2.2+cpu\\libtorch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(my_task "my_task.cpp" "my_task.h")

target_include_directories(my_task PRIVATE ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(my_task ${TORCH_LIBRARIES} ${OpenCV_LIBS})

if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET my_task
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:my_task>)
endif()

set_property(TARGET my_task PROPERTY CXX_STANDARD 20)
