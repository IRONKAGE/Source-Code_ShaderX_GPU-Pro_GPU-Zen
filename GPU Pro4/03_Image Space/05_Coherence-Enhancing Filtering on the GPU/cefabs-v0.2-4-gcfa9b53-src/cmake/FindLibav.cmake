#  LIBAV_FOUND - system has Libav libraries
#  LIBAV_INCLUDE_DIR - the Libav include directory
#  LIBAV_LIBRARIES - link these to use Libav

IF(WIN32)
    FIND_PATH( LIBAV_DIR include/libavformat/avformat.h PATH_SUFFIXES .. )
    GET_FILENAME_COMPONENT(LIBAV_DIR ${LIBAV_DIR} ABSOLUTE )
    SET(LIBAV_DIR ${LIBAV_DIR} CACHE FILEPATH "" FORCE)

    FIND_PATH( LIBAV_INCLUDE_DIR libavformat/avformat.h HINTS ${LIBAV_DIR}/include )

    FIND_LIBRARY( LIBAV_avutil_LIBRARY avutil HINTS ${LIBAV_DIR}/lib )
    FIND_LIBRARY( LIBAV_avcodec_LIBRARY avcodec HINTS ${LIBAV_DIR}/lib )
    FIND_LIBRARY( LIBAV_avformat_LIBRARY avformat HINTS ${LIBAV_DIR}/lib )
    FIND_LIBRARY( LIBAV_swscale_LIBRARY swscale HINTS ${LIBAV_DIR}/lib )

    FIND_PACKAGE_HANDLE_STANDARD_ARGS(LIBAV DEFAULT_MSG 
        LIBAV_INCLUDE_DIR
        LIBAV_avutil_LIBRARY
        LIBAV_avcodec_LIBRARY
        LIBAV_avformat_LIBRARY
        LIBAV_swscale_LIBRARY )

    IF(LIBAV_FOUND)
        SET(LIBAV_LIBRARIES
            ${LIBAV_avcodec_LIBRARY}
            ${LIBAV_avformat_LIBRARY}
            ${LIBAV_avutil_LIBRARY}
            ${LIBAV_swscale_LIBRARY} )
    ENDIF()
ELSE()
    INCLUDE(FindPkgConfig)
    PKG_CHECK_MODULES( LIBAV_PKGCONF QUIET libavutil libavcodec libavformat libswscale )

    FIND_PATH(LIBAV_INCLUDE_DIR libavformat/avformat.h HINTS ${LIBAV_PKGCONG_INCLUDE_DIRS})
    
    FIND_LIBRARY( LIBAV_avutil_LIBRARY avutil HINTS ${LIBAV_PKGCONF_LIBRARY_DIRS} )
    FIND_LIBRARY( LIBAV_avcodec_LIBRARY avcodec HINTS ${LIBAV_PKGCONF_LIBRARY_DIRS} )
    FIND_LIBRARY( LIBAV_avformat_LIBRARY avformat HINTS ${LIBAV_PKGCONF_LIBRARY_DIRS} )
    FIND_LIBRARY( LIBAV_swscale_LIBRARY swscale HINTS ${LIBAV_PKGCONF_LIBRARY_DIRS} )

    FIND_PACKAGE_HANDLE_STANDARD_ARGS(LIBAV DEFAULT_MSG 
        LIBAV_INCLUDE_DIR
        LIBAV_avcodec_LIBRARY
        LIBAV_avformat_LIBRARY
        LIBAV_avutil_LIBRARY
        LIBAV_swscale_LIBRARY )

    IF(LIBAV_FOUND)
        SET(LIBAV_LIBRARIES
            ${LIBAV_avcodec_LIBRARY}
            ${LIBAV_avformat_LIBRARY}
            ${LIBAV_avutil_LIBRARY}
            ${LIBAV_swscale_LIBRARY}
            ${LIBAV_PKGCONF_LDFLAGS_OTHER} )
    ENDIF()
ENDIF()

