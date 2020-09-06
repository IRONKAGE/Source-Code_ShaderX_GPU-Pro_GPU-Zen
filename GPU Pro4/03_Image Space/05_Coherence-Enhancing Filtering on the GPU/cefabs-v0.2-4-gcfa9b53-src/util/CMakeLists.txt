INCLUDE(${QT_USE_FILE})

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/gpu
    ${PROJECT_BINARY_DIR} )

FILE(GLOB sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.h *.ui)
IF(LIBAV_FOUND)
    INCLUDE_DIRECTORIES( ${LIBAV_INCLUDE_DIR} )
ELSE()
    LIST(REMOVE_ITEM sources 
        libav_decoder.cpp
        libav_decoder.h
        libav_encoder.cpp
        libav_encoder.h )
ENDIF()

QT4_AUTO_WRAP( sources ${sources} )
CUDA_ADD_LIBRARY( util STATIC ${sources} )

SOURCE_GROUP( src REGULAR_EXPRESSION "c$|cpp$|hpp$|h$|ui$|qrc$|cu$" )
SOURCE_GROUP( generated REGULAR_EXPRESSION "cxx$|ui_" )

IF(MSVC)
    IF(MSVC_IDE)
        SET_TARGET_PROPERTIES( util PROPERTIES COMPILE_FLAGS "/FIstable.h /Yustable.h" )
        SET_SOURCE_FILES_PROPERTIES( stable.cpp PROPERTIES COMPILE_FLAGS "/Ycstable.h" )
    ELSE()
        SET_TARGET_PROPERTIES( util PROPERTIES COMPILE_FLAGS "/FIstable.h" )
    ENDIF()
ELSE(MSVC)
    SET_TARGET_PROPERTIES( util PROPERTIES COMPILE_FLAGS "-include stable.h" )
ENDIF(MSVC)
