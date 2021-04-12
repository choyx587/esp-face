idf_build_get_property(target IDF_TARGET)

set(lib     libcat_face_detector.a
            libhuman_face_detector.a
            libdl.a)

set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/lib/esp32s3")
if(${IDF_TARGET} STREQUAL "esp32s3")
    set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/lib/esp32s3")
endif()

set(COMPONENT_ADD_INCLUDEDIRS   include/dl
                                include/model)

                                
target_link_libraries(${COMPONENT_TARGET}   INTERFACE   ${links})
target_link_libraries(${COMPONENT_TARGET}   INTERFACE   ${lib})

register_component()
