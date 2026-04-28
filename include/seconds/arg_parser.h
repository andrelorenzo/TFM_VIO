#ifndef PARSER_H_
#define PARSER_H_

#include "stdbool.h"
#include "stdint.h"

#ifndef FLAG_LIST_INIT_CAP
#define FLAG_LIST_INIT_CAP 1024
#endif // FLAG_LIST_INIT

typedef struct {
    const char *items[FLAG_LIST_INIT_CAP];
    size_t count;
} flag_list_t;


bool        *FlagBool  (const char * name,bool is_mandatory, bool def_val     , const char * desc);
uint8_t     *FlagUint8 (const char * name,bool is_mandatory, uint8_t   def_val, const char * desc);
uint16_t    *FlagUint16(const char * name,bool is_mandatory, uint16_t  def_val, const char * desc);
uint64_t    *FlagUint64(const char * name,bool is_mandatory, uint64_t  def_val, const char * desc);
uint32_t    *FlagUint32(const char * name,bool is_mandatory, uint32_t  def_val, const char * desc);
int         *FlagInt   (const char * name,bool is_mandatory, int       def_val, const char * desc);
float       *FlagFloat (const char * name,bool is_mandatory, float     def_val, const char * desc);
double      *FlagDouble(const char * name,bool is_mandatory, double    def_val, const char * desc);
char        **FlagStr  (const char * name,bool is_mandatory, char *    def_val, const char * desc);
size_t      *FlagSize  (const char * name,bool is_mandatory, size_t    def_val, const char * desc);
flag_list_t *FlagList  (const char * name,bool is_mandatory                   , const char * desc); 

bool        FlagParse(int argc, char ** argv);
void        FlagPrintHelp(FILE * stream);
void        FlagPrintError(FILE * stream);

int         FlagRestArgc(void);
char **     FlagRestArgv(void);
const char *FlagProgramName(void);
char *      FlagName(void *val);

#ifndef FLAG_ASSERT
#define FLAG_ASSERT(b) assert(b)    
#endif // FLAG_ASSERT

#ifndef UNUSED_VAR
#define UNUSED_VAR(a) (void)(a)
#endif // UNUSED_VAR

#ifndef UNUSED_FN
#define UNUSED_FN (void)
#endif // UNUSED_FN

#ifndef ARRAY_LEN
#define ARRAY_LEN(arr) (sizeof(arr)/sizeof((arr)[0]))
#endif // ARRAY_LEN

#ifndef FLAGS_CAP
#define FLAGS_CAP 256
#endif // FLAGS_CAP




#ifdef PARSER_IMP

#include "stddef.h"
#include "assert.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "limits.h"
#include "errno.h"
#include <ctype.h>

typedef enum {
    FLAG_BOOL = 0,
    FLAG_UINT8,
    FLAG_UINT16,
    FLAG_UINT32,
    FLAG_UINT64,
    FLAG_INT,
    FLAG_FLOAT,
    FLAG_DOUBLE,
    FLAG_SIZE,
    FLAG_STR,
    FLAG_LIST,
    COUNT_FLAG_TYPES,
} flag_type_e;

typedef union {
    bool        as_bool;
    uint8_t     as_uint8;
    uint16_t    as_uint16;
    uint32_t    as_uint32;
    uint64_t    as_uint64;
    int         as_int;
    float       as_float;
    double      as_double;
    char       *as_str;
    size_t      as_size;
    flag_list_t as_list;
} flag_val_u;

typedef enum {
    FLAG_NO_ERROR = 0,
    FLAG_ERROR_UNKNOWN,
    FLAG_ERROR_NO_VALUE,
    FLAG_ERROR_INVALID_NUMBER,
    FLAG_ERROR_INTEGER_OVERFLOW,
    FLAG_ERROR_FLOAT_OVERFLOW,
    FLAG_ERROR_DOUBLE_OVERFLOW,
    FLAG_ERROR_INVALID_SIZE_SUFFIX,
    COUNT_FLAG_ERRORS,
} flag_error_e;

typedef struct {
    flag_type_e type;
    const char *name;
    const char *desc;
    flag_val_u val;
    flag_val_u def;
    bool is_mandatory;
    bool has_changed;
} flag_t;

typedef struct {
    flag_t flags[FLAGS_CAP];
    size_t flags_count;

    flag_error_e flag_error;
    char *flag_error_name;
    char *flag_error_value;

    const char *program_name;

    int rest_argc;
    char **rest_argv;
} flag_ctx_t;


static flag_ctx_t flag_ctx;


void context_reset(){
    flag_ctx.flags_count = 0;
    flag_ctx.rest_argc = 0;
    flag_ctx.program_name = NULL;
    
    for(size_t i = 0; i < FLAGS_CAP; i++){
        flag_ctx.flags[i].has_changed = false;
        flag_ctx.flags[i].is_mandatory = false;
    }
}

static void flag_list_append(flag_t * f, char * item){
    size_t count = f->val.as_list.count;
    f->val.as_list.items[count] = item;
    f->val.as_list.count++;
}

static char *flag_shift_args(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    *argv += 1;
    *argc -= 1;
    return result;
}

static flag_t * flag_new_flag_(flag_ctx_t * ctx, flag_type_e _type, const char * _name, const char * _desc, bool _is_mandatory){
    FLAG_ASSERT(ctx->flags_count < FLAGS_CAP);
    flag_t * f =  &ctx->flags[ctx->flags_count++];
    memset(f, 0, sizeof(*f));

    f->type = _type;
    f->name = _name;
    f->desc = _desc;
    f->is_mandatory = _is_mandatory;

    return f;
}

///=======================================BOOL=======================================   
static bool * flag_new_bool_(flag_ctx_t * ctx, const char * _name, bool _def, bool is_mandatory, const char * _desc){
    flag_t * f = flag_new_flag_(ctx, FLAG_BOOL, _name, _desc, is_mandatory);
    
    f->def.as_bool = _def;
    f->val.as_bool = _def;    
    return &f->val.as_bool;
}

bool *FlagBool  (const char * name,bool is_mandatory, bool def_val, const char * desc){
    return flag_new_bool_(&flag_ctx, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================UINT8=======================================   
static uint8_t * flag_new_uint8_(flag_ctx_t * ctx, const char * _name, uint8_t _def, bool is_mandatory, const char * _desc){
    flag_t * f = flag_new_flag_(ctx, FLAG_UINT8, _name, _desc, is_mandatory);
    
    f->def.as_uint8 = _def;
    f->val.as_uint8 = _def;    
    return &f->val.as_uint8;
}

uint8_t *FlagUint8  (const char * name,bool is_mandatory, uint8_t def_val, const char * desc){
    return flag_new_uint8_(&flag_ctx, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================UINT16=======================================   
static uint16_t * flag_new_uint16_(flag_ctx_t * ctx, const char * _name, uint16_t _def, bool is_mandatory, const char * _desc){
    flag_t * f = flag_new_flag_(ctx, FLAG_UINT16, _name, _desc, is_mandatory);
    
    f->def.as_uint16 = _def;
    f->val.as_uint16 = _def;    
    return &f->val.as_uint16;
}

uint16_t *FlagUint16  (const char * name, bool is_mandatory, uint16_t def_val, const char * desc){
    return flag_new_uint16_(&flag_ctx, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================UINT32=======================================   
static uint32_t * flag_new_uint32_(flag_ctx_t * ctx, const char * _name, uint32_t _def, bool is_mandatory, const char * _desc){
    flag_t * f = flag_new_flag_(ctx, FLAG_UINT32, _name, _desc, is_mandatory);
    
    f->def.as_uint32 = _def;
    f->val.as_uint32 = _def;    
    return &f->val.as_uint32;
}

uint32_t *FlagUint32  (const char * name,bool is_mandatory, uint32_t def_val, const char * desc){
    return flag_new_uint32_(&flag_ctx, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================UINT64=======================================   
static uint64_t * flag_new_uint64_(flag_ctx_t * ctx, const char * _name, uint64_t _def, bool is_mandatory, const char * _desc){
    flag_t * f = flag_new_flag_(ctx, FLAG_UINT64, _name, _desc, is_mandatory);
    
    f->def.as_uint64 = _def;
    f->val.as_uint64 = _def;    
    return &f->val.as_uint64;
}

uint64_t *FlagUint64  (const char * name,bool is_mandatory, uint64_t def_val, const char * desc){
    return flag_new_uint64_(&flag_ctx, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================INT=======================================   
static int * flag_new_int_(flag_ctx_t * ctx, const char * _name, int _def, bool is_mandatory, const char * _desc){
    flag_t * f = flag_new_flag_(ctx, FLAG_INT, _name, _desc, is_mandatory);
    
    f->def.as_int = _def;
    f->val.as_int = _def;
    return &f->val.as_int;
}

int *FlagInt  (const char * name,bool is_mandatory, int def_val, const char * desc){
    return flag_new_int_(&flag_ctx, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================FLOAT=======================================   
static float * flag_new_float_(flag_ctx_t * ctx, const char * _name, float _def, bool is_mandatory, const char * _desc){
    flag_t * f = flag_new_flag_(ctx, FLAG_FLOAT, _name, _desc, is_mandatory);
    
    f->def.as_float = _def;
    f->val.as_float = _def;    
    return &f->val.as_float;
}

float *FlagFloat  (const char * name,bool is_mandatory, float def_val, const char * desc){
    return flag_new_float_(&flag_ctx, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================DOUBLE=======================================   
static double * flag_new_double_(flag_ctx_t * ctx, const char * _name, double _def, bool is_mandatory, const char * _desc){
    flag_t * f = flag_new_flag_(ctx, FLAG_DOUBLE, _name, _desc, is_mandatory);
    
    f->def.as_double = _def;
    f->val.as_double = _def;    
    return &f->val.as_double;
}

double *FlagDouble  (const char * name,bool is_mandatory, double def_val, const char * desc){
    return flag_new_double_(&flag_ctx, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================SIZE=======================================   
static size_t * flag_new_size_(flag_ctx_t * ctx, const char * _name, size_t _def, bool is_mandatory, const char * _desc){
    flag_t * f = flag_new_flag_(ctx, FLAG_SIZE, _name, _desc, is_mandatory);
    
    f->def.as_size = _def;
    f->val.as_size = _def;    
    return &f->val.as_size;
}

size_t *FlagSize  (const char * name,bool is_mandatory, size_t def_val, const char * desc){
    return flag_new_size_(&flag_ctx, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================string=======================================   
static char ** flag_new_string_(flag_ctx_t * ctx, const char * _name, char * _def, bool is_mandatory , const char * _desc){
    flag_t * f = flag_new_flag_(ctx, FLAG_STR, _name, _desc, is_mandatory);
    
    f->def.as_str = _def;
    f->val.as_str = _def;
    return &f->val.as_str;
}

char **FlagStr  (const char * name,bool is_mandatory, char * def_val, const char * desc){
    return flag_new_string_(&flag_ctx, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================LIST=======================================   
static flag_list_t * flag_new_list_(flag_ctx_t * ctx, const char * _name, const char * _desc, bool is_mandatory ){
    flag_t * f = flag_new_flag_(ctx, FLAG_LIST, _name, _desc, is_mandatory);
    return &f->val.as_list;
}

flag_list_t *FlagList  (const char * name, bool is_mandatory, const char * desc){
    return flag_new_list_(&flag_ctx, name, desc, is_mandatory);
}
///==================================================================================


static void *flag_get_ref(flag_t *flag){
    return &flag->val;
}

int FlagRestArgc(void){
    return flag_ctx.rest_argc;
}
char ** FlagRestArgv(void){
    return flag_ctx.rest_argv;
}
const char * FlagProgramName(void){
    return flag_ctx.program_name;
}

char *flag_name(flag_ctx_t * ctx, void *val){

    for(size_t i = 0; i < ctx->flags_count; i++){
        flag_t * f = &ctx->flags[i];
        if(flag_get_ref(f) == val){
            return (char*)f->name;
        }
    }
    return NULL;
}

char * FlagName(void *val){
    return flag_name(&flag_ctx, val);
}


bool flag_parse(flag_ctx_t * c, int argc, char ** argv){
    // context_reset();

    if(c->program_name == NULL){
        c->program_name = flag_shift_args(&argc, &argv);
    }
    bool mandatory_failed = false;
    while (argc > 0) {
        char *flag = flag_shift_args(&argc, &argv);
        if (*flag != '-' || strcmp(flag, "--") == 0) {
            c->rest_argc = argc + 1;
            c->rest_argv = argv - 1;
            return true;
            fprintf(stdout,"%s", flag);
        }

        flag += 1; // remove the '-'
        bool ignore = false;
        if (*flag == '/') {
            ignore = true;
            flag += 1;
        }

        char * equals = strchr(flag, '=');
        if (equals != NULL) {
            *equals = '\0';
            equals += 1; // pointer to the actual value
        }
        bool found = false;

        if(strcmp("-h", flag) == 0 || strcmp("-help", flag) == 0){
            system("clear");

            FlagPrintHelp(stdout);

            return true;
        }

        for (size_t i = 0; i < c->flags_count; ++i) {
            if (strcmp(c->flags[i].name, flag) == 0) {
                switch (c->flags[i].type){
                    case FLAG_BOOL: {

                        char * arg;
                        if(equals == NULL){
                            if(argc == 0){
                                c->flag_error = FLAG_ERROR_NO_VALUE;
                                c->flag_error_name = flag;
                                return false;
                            }
                            arg = flag_shift_args(&argc, &argv);
                        }else{
                            arg = equals;
                        }

                        if(!ignore){
                            if( strcmp(arg, "1") == 0 ||  strcmp(arg, "true") == 0  || strcmp(arg, "yes") == 0 || strcmp(arg, "si") == 0) {
                                c->flags[i].val.as_bool = true;
                            }else if (strcmp(arg, "0") == 0 ||  strcmp(arg, "false") == 0  || strcmp(arg, "no") == 0){
                                c->flags[i].val.as_bool = false;
                            }else{ 
                                c->flag_error = FLAG_ERROR_UNKNOWN;
                                c->flag_error_name = flag;
                                return false;
                            }
                        }
                        c->flags[i].has_changed = true;
                    break;
                    }
                    case FLAG_UINT8: {
                        char * arg;
                        if(equals == NULL){
                            if(argc == 0){
                                c->flag_error = FLAG_ERROR_NO_VALUE;
                                c->flag_error_name = flag;
                                return false;
                            }
                            arg = flag_shift_args(&argc, &argv);
                        }else{
                            arg = equals;
                        }
                        errno = 0;
                        char * ptr;
                        static_assert(sizeof(unsigned char) == sizeof(uint8_t), "size mismatch in uint8_t , uint8_t muust be typedef to unsigned char");
                        unsigned long long tmp = strtoull(arg,&ptr,10);
                        if(*ptr != '\0'){
                            c->flag_error = FLAG_ERROR_INVALID_NUMBER;
                            c->flag_error_name = flag;
                            return false;
                        }
                        if ((tmp == ULLONG_MAX && errno == ERANGE) || tmp > UINT8_MAX) {
                            c->flag_error = FLAG_ERROR_INTEGER_OVERFLOW;
                            c->flag_error_name = flag;
                            return false;
                        }
                        if (!ignore) {
                            c->flags[i].val.as_uint8 = (uint8_t)tmp;
                        }
                        c->flags[i].has_changed = true;
                    break;
                    }
                    case FLAG_UINT16: {
                        char * arg;
                        if(equals == NULL){
                            if(argc == 0){
                                c->flag_error = FLAG_ERROR_NO_VALUE;
                                c->flag_error_name = flag;
                                return false;
                            }
                            arg = flag_shift_args(&argc, &argv);
                        }else{
                            arg = equals;
                        }
                        errno = 0;
                        char * ptr;
                        static_assert(sizeof(unsigned short int) == sizeof(uint16_t), "size mismatch in uint16_t , uint16_t muust be typedef to unsigned short int");
                        unsigned long long tmp = strtoull(arg,&ptr,10);
                        if(*ptr != '\0'){
                            c->flag_error = FLAG_ERROR_INVALID_NUMBER;
                            c->flag_error_name = flag;
                            return false;
                        }
                        if ((tmp == ULLONG_MAX && errno == ERANGE) || tmp > UINT16_MAX) {
                            c->flag_error = FLAG_ERROR_INTEGER_OVERFLOW;
                            c->flag_error_name = flag;
                            return false;
                        }
                        if (!ignore) {
                            c->flags[i].val.as_uint16 = (uint16_t)tmp;
                        }
                        c->flags[i].has_changed = true;
                    break;
                    }
                    case FLAG_UINT32: {
                        char * arg;
                        if(equals == NULL){
                            if(argc == 0){
                                c->flag_error = FLAG_ERROR_NO_VALUE;
                                c->flag_error_name = flag;
                                return false;
                            }
                            arg = flag_shift_args(&argc, &argv);
                        }else{
                            arg = equals;
                        }
                        errno = 0;
                        char * ptr;
                        static_assert(sizeof(unsigned int) == sizeof(uint32_t), "size mismatch in uint32_t , uint32_t muust be typedef to unsigned int");
                        unsigned long long tmp = strtoull(arg,&ptr,10);
                        if(*ptr != '\0'){
                            c->flag_error = FLAG_ERROR_INVALID_NUMBER;
                            c->flag_error_name = flag;
                            return false;
                        }
                        if ((tmp == ULLONG_MAX && errno == ERANGE) || tmp > UINT32_MAX) {
                            c->flag_error = FLAG_ERROR_INTEGER_OVERFLOW;
                            c->flag_error_name = flag;
                            return false;
                        }
                        if (!ignore) {
                            c->flags[i].val.as_uint32 = (uint32_t)tmp;
                        }
                        c->flags[i].has_changed = true;
                    break;
                    }
                    case FLAG_UINT64: {
                        char * arg;
                        if(equals == NULL){
                            if(argc == 0){
                                c->flag_error = FLAG_ERROR_NO_VALUE;
                                c->flag_error_name = flag;
                                return false;
                            }
                            arg = flag_shift_args(&argc, &argv);
                        }else{
                            arg = equals;
                        }
                        errno = 0;
                        char * ptr;
                        // static_assert(sizeof(unsigned long int) == sizeof(uint64_t), "size mismatch in uint64_t , uint64_t muust be typedef to unsigned long int");
                        unsigned long long tmp = strtoull(arg,&ptr,10);
                        if(*ptr != '\0'){
                            c->flag_error = FLAG_ERROR_INVALID_NUMBER;
                            c->flag_error_name = flag;
                            return false;
                        }
                        /* strtoull() already returns unsigned long long; bounds-check to 64 bits */
                        if ((tmp == ULLONG_MAX && errno == ERANGE) || tmp > UINT64_MAX) {
                            c->flag_error = FLAG_ERROR_INTEGER_OVERFLOW;
                            c->flag_error_name = flag;
                            return false;
                        }

                        if (!ignore) {
                            c->flags[i].val.as_uint64 = (uint64_t)tmp;
                        }
                        c->flags[i].has_changed = true;
                    break;
                    }
                    case FLAG_INT: {
                        char * arg;
                        if(equals == NULL){
                            if(argc == 0){
                                c->flag_error = FLAG_ERROR_NO_VALUE;
                                c->flag_error_name = flag;
                                return false;
                            }
                            arg = flag_shift_args(&argc, &argv);
                        }else{
                            arg = equals;
                        }
                        errno = 0;
                        char * ptr;
                        long v = strtol(arg, &ptr, 10);
                        if(*ptr != '\0'){
                            c->flag_error = FLAG_ERROR_INVALID_NUMBER;
                            c->flag_error_name = flag;
                            return false;
                        }
                        if ((v == LONG_MAX || v == LONG_MIN) && errno == ERANGE) {
                            c->flag_error = FLAG_ERROR_INTEGER_OVERFLOW;
                            c->flag_error_name = flag;
                            return false;
                        }
                        if (v > INT_MAX || v < INT_MIN){
                            c->flag_error = FLAG_ERROR_INTEGER_OVERFLOW;
                            c->flag_error_name = flag;
                            return false;
                        }
                        if(!ignore){
                            c->flags[i].val.as_int = (int)v;
                        }
                        c->flags[i].has_changed = true;
                    break;
                    }
                    case FLAG_FLOAT: {
                        char * arg;
                        if(equals == NULL){
                            if(argc == 0){
                                c->flag_error = FLAG_ERROR_NO_VALUE;
                                c->flag_error_name = flag;
                                return false;
                            }
                            arg = flag_shift_args(&argc, &argv);
                        }else{
                            arg = equals;
                        }
                        errno = 0;
                        char * ptr;
                        float f = strtof(arg, &ptr);
                        if(*ptr != '\0' && *ptr != '\0'){
                            c->flag_error = FLAG_ERROR_INVALID_NUMBER;
                            c->flag_error_name = flag;
                            return false;
                        }
                        if ((f == FLT_MAX || f == -FLT_MAX) && errno == ERANGE) {
                            c->flag_error = FLAG_ERROR_FLOAT_OVERFLOW;
                            c->flag_error_name = flag;
                            return false;
                        }
                        if (!ignore) {
                            c->flags[i].val.as_float = f;
                        }
                        c->flags[i].has_changed = true;
                    break;
                    }
                    case FLAG_DOUBLE: {
                        char * arg;
                        if(equals == NULL){
                            if(argc == 0){
                                c->flag_error = FLAG_ERROR_NO_VALUE;
                                c->flag_error_name = flag;
                                return false;
                            }
                            arg = flag_shift_args(&argc, &argv);
                        }else{
                            arg = equals;
                        }
                        errno = 0;
                        char * ptr;
                        double d = strtod(arg, &ptr);
                        if(*ptr != '\0'){
                            c->flag_error = FLAG_ERROR_INVALID_NUMBER;
                            c->flag_error_name = flag;
                            return false;
                        }
                        if ((d == DBL_MAX || d == -DBL_MAX) && errno == ERANGE) {
                            c->flag_error = FLAG_ERROR_FLOAT_OVERFLOW;
                            c->flag_error_name = flag;
                            return false;
                        }
                        if (!ignore) {
                            c->flags[i].val.as_double = d;
                        }
                        c->flags[i].has_changed = true;
                    break;
                    }
                    case FLAG_SIZE: {
                        char * arg;
                        if(equals == NULL){
                            if(argc == 0){
                                c->flag_error = FLAG_ERROR_NO_VALUE;
                                c->flag_error_name = flag;
                                return false;
                            }
                            arg = flag_shift_args(&argc, &argv);
                        }else{
                            arg = equals;
                        }
                        // Parse size with optional suffix: K/M/G/T/P (powers of 1024)
                        errno = 0;
                        char * ptr;
                        unsigned long long base = strtoull(arg, &ptr, 10);
                        if ((base == ULLONG_MAX && errno == ERANGE)) {
                            c->flag_error = FLAG_ERROR_INTEGER_OVERFLOW;
                            c->flag_error_name = flag;
                            return false;
                        }
                        unsigned long long mult = 1;
                        if (*ptr != '\0') {
                            char s = *ptr;
                            char s2 = (char)toupper((unsigned char)s);
                            ptr++;
                            if (*ptr == '\0') {
                                switch (s2) {
                                    case 'K': mult = 1024ull; break;
                                    case 'M': mult = 1024ull*1024ull; break;
                                    case 'G': mult = 1024ull*1024ull*1024ull; break;
                                    case 'T': mult = 1024ull*1024ull*1024ull*1024ull; break;
                                    case 'P': mult = 1024ull*1024ull*1024ull*1024ull*1024ull; break;
                                    default: {
                                        c->flag_error = FLAG_ERROR_INVALID_NUMBER;
                                        c->flag_error_name = flag;
                                        return false;
                                    }
                                }
                            } else {
                                // Trailing garbage (e.g., "10MB" where we only allow single-letter suffix)
                                c->flag_error = FLAG_ERROR_INVALID_NUMBER;
                                c->flag_error_name = flag;
                                return false;
                            }
                        }
                        // Overflow check for size_t range
                        // __uint128_t wide = (__uint128_t)base * (__uint128_t)mult;
                        // if (wide > ( __uint128_t)SIZE_MAX) {
                        //     c->flag_error = FLAG_ERROR_INTEGER_OVERFLOW;
                        //     c->flag_error_name = flag;
                        //     return false;
                        // }
                        if (!ignore) {
                            c->flags[i].val.as_size = (size_t)( (unsigned long long)(base * mult) );
                        }
                        c->flags[i].has_changed = true;
                    break;
                    }
                    case FLAG_STR: {
                        char * arg;
                        if(equals == NULL){
                            if(argc == 0){
                                c->flag_error = FLAG_ERROR_NO_VALUE;
                                c->flag_error_name = flag;
                                return false;
                            }
                            arg = flag_shift_args(&argc, &argv);
                        }else{
                            arg = equals;
                        }
                        if (!ignore) {
                            c->flags[i].val.as_str = arg; // guarda el puntero; copia si necesitas propiedad
                        }
                        c->flags[i].has_changed = true;
                    break;
                    }
                    case FLAG_LIST: {
                        // TODO: implementar según la representación de lista en tu union/struct.
                        // Por ahora tratamos cada ocurrencia como string simple.
                        char * arg;
                        if(equals == NULL){
                            if(argc == 0){
                                c->flag_error = FLAG_ERROR_NO_VALUE;
                                c->flag_error_name = flag;
                                return false;
                            }
                            arg = flag_shift_args(&argc, &argv);
                        }else{
                            arg = equals;
                        }
                        if (!ignore) {
                            // Fallback: si no hay contenedor de lista, al menos guarda el último valor como cadena.
                            flag_list_append(&c->flags[i], arg);
                        }
                        c->flags[i].has_changed = true;
                    break;
                    }
                    case COUNT_FLAG_TYPES:
                    default: {
                        assert(0 && "unreachable");
                        exit(69);
                    }
                }
                found = true;
            }
        }

        if (!found) {
            c->flag_error = FLAG_ERROR_UNKNOWN;
            c->flag_error_name = flag;
            c->flag_error_value = flag;
            return false;
        }
    } 
    for (size_t i = 0; i < c->flags_count; ++i) {
        if(c->flags[i].is_mandatory && !c->flags[i].has_changed){
            c->flag_error = FLAG_ERROR_NO_VALUE;
            c->flag_error_name = (char*)c->flags[i].name;
            FlagPrintError(stdout);
            mandatory_failed = true;
        }
    }
    if(mandatory_failed){
        // FlagPrintHelp(stdout);
        return false;
    }
    c->rest_argc = argc;
    c->rest_argv = argv;
    return true;
}

bool FlagParse(int argc, char ** argv){
    return flag_parse(&flag_ctx, argc, argv);
}

void FlagPrintHelp(FILE *stream){

    static const char * def_usage =     "Usage: %s [OPTIONS]\n"
                                        "\n"
                                        "OPTIONS:\n"
                                        "\n"
                                        "    DEFAULT FLAGS:\n"
                                        "    -f, -file <str>\n"
                                        "        File name to be parsed\n"
                                        "        Default: NONE\n"
                                        "    -h, -help <bool>\n"
                                        "        Show this message\n"
                                        "        Default: false\n"
                                        "    CUSTOM FLAGS:\n";

    fprintf(stream, def_usage, flag_ctx.program_name);

    for(size_t i = 0; i < flag_ctx.flags_count; i++){
        flag_t * f = &flag_ctx.flags[i]; 
        switch(flag_ctx.flags[i].type){
            case FLAG_BOOL : {
                fprintf(stream,"    -%s <bool>\n", f->name);
                fprintf(stream,"        %s\n", f->desc);
                if(!f->is_mandatory){
                    fprintf(stream,"        Default: %s\n", f->def.as_bool ? "true" : "false");
                }else{
                    fprintf(stream, "        MANDATORY\n");
                }
            break; 
            }  
            case FLAG_UINT8 : {
                fprintf(stream,"    -%s <uint8>\n", f->name);
                fprintf(stream,"        %s\n", f->desc);
                if(!f->is_mandatory){
                    fprintf(stream,"        Default: %u\n", f->def.as_uint8);
                }else{
                    fprintf(stream, "        MANDATORY\n");
                }
            break; 
            }  
            case FLAG_UINT16 : {
                fprintf(stream,"    -%s <uint16>\n", f->name);
                fprintf(stream,"        %s\n", f->desc);
                if(!f->is_mandatory){
                    fprintf(stream,"        Default: %u\n", f->def.as_uint16);
                }else{
                    fprintf(stream, "        MANDATORY\n");
                }
            break;    
            }  
            case FLAG_UINT32 : {
                fprintf(stream,"    -%s <uint32>\n", f->name);
                fprintf(stream,"        %s\n", f->desc);
                if(!f->is_mandatory){
                    fprintf(stream,"        Default: %u\n", f->def.as_uint32);
                }else{
                    fprintf(stream, "        MANDATORY\n");
                }
            break;    
            }  
            case FLAG_UINT64 : {
                fprintf(stream,"    -%s <uint64>\n", f->name);
                fprintf(stream,"        %s\n", f->desc);
                if(!f->is_mandatory){
                    fprintf(stream,"        Default: %llu\n", f->def.as_uint64);
                }else{
                    fprintf(stream, "        MANDATORY\n");
                }
            break;    
            }  
            case FLAG_INT : {
                fprintf(stream,"    -%s <int>\n", f->name);
                fprintf(stream,"        %s\n", f->desc);
                if(!f->is_mandatory){
                    fprintf(stream,"        Default: %i\n", f->def.as_int);
                }else{
                    fprintf(stream, "        MANDATORY\n");
                }
            break;   
            }  
            case FLAG_FLOAT : {
                fprintf(stream,"    -%s <float>\n", f->name);
                fprintf(stream,"        %s\n", f->desc);
                if(!f->is_mandatory){
                    fprintf(stream,"        Default: %.4f\n", f->def.as_float);
                }else{
                    fprintf(stream, "        MANDATORY\n");
                }
            break; 
            }  
            case FLAG_DOUBLE : {
                fprintf(stream,"    -%s <double>\n", f->name);
                fprintf(stream,"        %s\n", f->desc);
                if(!f->is_mandatory){
                    fprintf(stream,"        Default: %.8f\n", f->def.as_double);
                }else{
                    fprintf(stream, "        MANDATORY\n");
                }
            break;    
            }  
            case FLAG_SIZE : {
                fprintf(stream,"    -%s <size>\n", f->name);
                fprintf(stream,"        %s\n", f->desc);
                if(!f->is_mandatory){
                    fprintf(stream,"        Default: %llu\n", f->def.as_size);
                }else{
                    fprintf(stream, "        MANDATORY\n");
                }
            break;  
            }  
            case FLAG_STR : {
                fprintf(stream,"    -%s <str>\n", f->name);
                fprintf(stream,"        %s\n", f->desc);
                if(!f->is_mandatory){
                    fprintf(stream,"        Default: %s\n", f->def.as_str);
                }else{
                    fprintf(stream, "        MANDATORY\n");
                }
            break;   
            }  
            case FLAG_LIST : {
                fprintf(stream,"    -%s <str> ... %s <str> ...\n", f->name, f->name);
                fprintf(stream,"        %s\n", f->desc);
            break;  
            }  
            default:
                assert(0 && "unreachable");
                exit(-1);
            break;
        }
    }


}
void FlagPrintError(FILE * stream){
    flag_ctx_t * fc = &flag_ctx;
    switch(fc->flag_error){
        case FLAG_NO_ERROR : {
            fprintf(stream,"Task failed succesfully, No error :/\n");
        break;
        }
        case FLAG_ERROR_UNKNOWN : {
            fprintf(stream,"ERROR: -%s: unknown flag\n", fc->flag_error_name);
            fprintf(stream,"    %s is not a valid flag\n", fc->flag_error_value);
        break;
        }
        case FLAG_ERROR_NO_VALUE : {
            fprintf(stream,"ERROR: -%s: no value provided\n", fc->flag_error_name);
        break;
        }
        case FLAG_ERROR_INVALID_NUMBER : {
            fprintf(stream,"ERROR: -%s: invalid number\n", fc->flag_error_name);
        break;
        }
        case FLAG_ERROR_INTEGER_OVERFLOW : {
            fprintf(stream,"ERROR: -%s: integer overflow\n", fc->flag_error_name);
        break;
        }
        case FLAG_ERROR_FLOAT_OVERFLOW : {
            fprintf(stream,"ERROR: -%s: float overflow\n", fc->flag_error_name);
        break;
        }
        case FLAG_ERROR_DOUBLE_OVERFLOW : {
            fprintf(stream,"ERROR: -%s: double overflow\n", fc->flag_error_name);
        break;
        }
        case FLAG_ERROR_INVALID_SIZE_SUFFIX : {
            fprintf(stream, "ERROR: -%s: invalid size suffix\n", fc->flag_error_name);
            fprintf(stream, "    Got %s suffix which is not expected\n", fc->flag_error_value);
        break;
        }
        default:
            assert(0 && "unreachable");
            exit(-1);
        break;
    } 
}
#endif // PARSER_IMP

#endif // PARSER_H_