#ifndef LOGGER_H_
#define LOGGER_H_


typedef enum{
    DEBUG = 0U,
    INFO,
    WARN,
    ERROR,
    FATAL,
    NONE
}verb_e;

static verb_e max_verbosity = INFO;

void Logger(verb_e verbosity, const char * format, ...);
verb_e LoggerGetVerbsity(void);
void LoggerSetVerbsity(verb_e);



#ifndef UNUSED_VAR
#define UNUSED_VAR(a) (void)(a)
#endif
#ifndef UNUSED_FN
#define UNUSED_FN (void)
#endif
#ifndef ARRAY_LEN
#define ARRAY_LEN(arr) (sizeof(arr)/sizeof((arr)[0]))
#endif

#ifdef LOGGER_IMP

#if defined(_WIN32) || defined(__unix__) || defined(__APPLE__)
#define LOGGER_WEAK __attribute__((weak))
#else
#define LOGGER_WEAK __weak
#endif

#include "stdarg.h"
#include "stdio.h"

/// @brief Get the actual verbosity level
/// @return the actual verbosity level
verb_e LoggerGetVerbsity(){
    return max_verbosity;
}
/// @brief Set the max verbosity level
/// @param max_verb max verbosity level, i.e if max_verb is set to WARN, only WARN, ERROR AND FATAL will be printed 
void LoggerSetVerbsity(verb_e max_verb){
    max_verbosity = max_verb;
}

/// @brief Function must be ovewritten with whatever implementation is of your need, some examples are provided
/// @param msg // msg to be send
/// @param size // size of the msg to me send
void printOut(verb_e verbosity, const char * msg, size_t size){
    const char *color = "";
    const char *tag   = "";

    switch(verbosity){
        case DEBUG: color = "\x1b[90m"; tag = "DEBUG"; break; // gris
        case INFO:  color = "\x1b[32m"; tag = "INFO "; break; // verde
        case WARN:  color = "\x1b[33m"; tag = "WARN "; break; // amarillo
        case ERROR: color = "\x1b[31m"; tag = "ERROR"; break; // rojo
        case FATAL: color = "\x1b[1;31m"; tag = "FATAL"; break; // rojo brillante/negrita
        case NONE:  return;
        default:    color = "\x1b[0m";  tag = "LOG  "; break;
    }

    const char *reset = "\x1b[0m";
    printf("%s[%s] %.*s%s\n", color, tag, (int)size, msg, reset);
    fflush(stdout);
}
/// @brief Actual logging function
/// @param verbosity verbosity of the message, never set it to NONE
/// @param format msg to be send
/// @param  variadic arguments of the function
void Logger(verb_e verbosity, const char * format, ...){

    if(verbosity >= max_verbosity){
        va_list args;
        va_start(args, format);
        char msg[1024];
        int size = vsprintf(msg, format, args);
        va_end(args);
        printOut(verbosity, msg, size);

    }

}
#endif // LOGGER_IMP

#endif