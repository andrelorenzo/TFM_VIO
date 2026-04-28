#ifndef FILE_PARSER_H_
#define FILE_PARSER_H_
#include "stdbool.h"
#include "stdint.h"


#ifndef MAX_BIN
#define MAX_BIN 2048
#endif // MAX_BIN

#ifndef PARAM_LIST_INIT_CAP
#define PARAM_LIST_INIT_CAP 1024
#endif // PARAM_LIST_INIT

typedef enum {
    PARAM_BOOL = 0,    
    PARAM_UINT,
    PARAM_INT,
    PARAM_FLOAT,
    PARAM_STR,
    PARAM_LIST,
    PARAM_BINARY,
    COUNT_PARAM_TYPES,
} param_type_e;

typedef union {
    bool        as_bool;
    uint32_t    as_uint;
    int         as_int;
    float       as_float;
    char       *as_str;
} param_simple_val_u;

typedef struct {
    param_simple_val_u items[PARAM_LIST_INIT_CAP];
    size_t count;
    param_type_e type;
} param_list_t;


typedef enum{
    FILE_TYPE_TXT = 0U,
    FILE_TYPE_CSV
}file_type_e;

typedef struct{
    char column_name[256];
    float column_item[MAX_BIN];
}csv_data_t;

typedef struct{
    csv_data_t csv_data[MAX_BIN];
    size_t item_count;
    size_t col_count;
    char *column_sep;
    char *decimal_sep;
}csv_fileh_t;


/* ARGUMENT BASED PARSER*/

void ParamBool  (bool * var, const char * name,bool is_mandatory, bool def_val     , const char * desc);
void ParamUint  (uint32_t * var, const char * name,bool is_mandatory, uint32_t  def_val, const char * desc);
void ParamInt   (int * var, const char * name,bool is_mandatory, int       def_val, const char * desc);
void ParamFloat (float * var, const char * name,bool is_mandatory, float     def_val, const char * desc);
void ParamStr   (char ** var, const char * name,bool is_mandatory, char *    def_val, const char * desc);
void ParamList  (param_list_t * var, const char * name, const char * desc, param_type_e type); 
void ParamBin   (uint8_t * var, size_t len, const char * name, const char * desc, uint8_t def_val);

bool InitCSV  (const char * dec_sep, const char * col_sep, csv_fileh_t * fileh);
int  GetCSVData (const char * name, float * data, size_t data_len);
bool AddRowCSV(float * row, size_t row_n);


bool  ParamSave(const char * filename, file_type_e type);
bool  ParamParse(const char * filename, file_type_e type);
void  ParamPrintError(FILE * stream);
char *ParamName(void *val);



#ifndef PARAM_ASSERT
#define PARAM_ASSERT(b) assert(b)    
#endif // PARAM_ASSERT

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

typedef union {
    param_simple_val_u simple_val;
    param_list_t list;
    uint8_t bin[MAX_BIN];
} param_val_u;

typedef enum {
    PARAM_NO_ERROR = 0,
    PARAM_ERROR_UNKNOWN,
    PARAM_ERROR_NO_VALUE,
    PARAM_ERROR_INVALID_NUMBER,
    PARAM_ERROR_INVALID_FILE_EXT,
    PARAM_ERROR_INVALID_SIZE_SUFFIX,
    PARAM_ERROR_BIN_OVERFLOW,
    COUNT_PARAM_ERRORS,
} param_error_e;

typedef struct {
    param_type_e type;
    param_type_e list_type;
    size_t list_bin_len;
    const char *name;
    const char *desc;
    void * ref;
    param_val_u def;


    bool is_mandatory;
    bool has_changed;
} param_t;



typedef struct {
    // txt params
    param_t params[FLAGS_CAP];
    size_t params_count;

    // csv data 
    csv_fileh_t * csv_fileh;

    param_error_e param_error;
} param_ctx_t;


static param_ctx_t param_ctx;


// void context_reset(){
//     param_ctx.params_count = 0;
    
//     for(size_t i = 0; i < FLAGS_CAP; i++){
//         param_ctx.params[i].has_changed = false;
//         param_ctx.params[i].is_mandatory = false;
//     }
// }


static param_t * param_new_param_(param_ctx_t * ctx, param_type_e _type, const char * _name, const char * _desc, bool _is_mandatory){
    PARAM_ASSERT(ctx->params_count < FLAGS_CAP);
    param_t * f =  &ctx->params[ctx->params_count++];
    memset(f, 0, sizeof(*f));

    f->type = _type;
    f->name = _name;
    f->desc = _desc;
    f->is_mandatory = _is_mandatory;

    return f;
}

///=======================================BOOL=======================================   
static void param_new_bool_(param_ctx_t * ctx, bool * var, const char * _name, bool _def, bool is_mandatory, const char * _desc){
    param_t * f = param_new_param_(ctx, PARAM_BOOL, _name, _desc, is_mandatory);
    
    f->def.simple_val.as_bool = _def;
    f->ref = var;    
    *var = _def;
}

void ParamBool  (bool * var, const char * name,bool is_mandatory, bool def_val, const char * desc){
    param_new_bool_(&param_ctx,var, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================UINT32=======================================   
static void param_new_uint32_(param_ctx_t * ctx,uint32_t * var, const char * _name, uint32_t _def, bool is_mandatory, const char * _desc){
    param_t * f = param_new_param_(ctx, PARAM_UINT, _name, _desc, is_mandatory);
    
    f->def.simple_val.as_uint = _def;
    f->ref =  var;    
    *var = _def;
}

void ParamUint  (uint32_t * var, const char * name,bool is_mandatory, uint32_t def_val, const char * desc){
    param_new_uint32_(&param_ctx,var, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================INT=======================================   
static void param_new_int_(param_ctx_t * ctx,int * var, const char * _name, int _def, bool is_mandatory, const char * _desc){
    param_t * f = param_new_param_(ctx, PARAM_INT, _name, _desc, is_mandatory);
    
    f->def.simple_val.as_int = _def;
    f->ref = var;
    *var = _def;
}

void ParamInt  (int * var, const char * name,bool is_mandatory, int def_val, const char * desc){
    param_new_int_(&param_ctx,var, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================FLOAT=======================================   
static void param_new_float_(param_ctx_t * ctx,float * var, const char * _name, float _def, bool is_mandatory, const char * _desc){
    param_t * f = param_new_param_(ctx, PARAM_FLOAT, _name, _desc, is_mandatory);
    
    f->def.simple_val.as_float = _def;
    f->ref = var;
    *var = _def;
}

void ParamFloat  (float * var, const char * name,bool is_mandatory, float def_val, const char * desc){
    param_new_float_(&param_ctx,var, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================string=======================================   
static void param_new_string_(param_ctx_t * ctx,char ** var, const char * _name, char * _def, bool is_mandatory , const char * _desc){
    param_t * f = param_new_param_(ctx, PARAM_STR, _name, _desc, is_mandatory);
    
    f->def.simple_val.as_str = _def;
    f->ref = var;
    *var = _def;
}

void ParamStr  (char ** var, const char * name,bool is_mandatory, char * def_val, const char * desc){
    param_new_string_(&param_ctx,var, name, def_val, is_mandatory, desc);
}
///==================================================================================

///=======================================LIST=======================================   
static void param_new_list_(param_ctx_t * ctx, param_list_t * var, const char * _name, const char * _desc, param_type_e type ){
    param_t * f = param_new_param_(ctx, PARAM_LIST, _name, _desc, true);
    
    f->ref = var;
    var->type = type;
    f->def.list.type = type;
}

void ParamList  (param_list_t * var, const char * name, const char * desc, param_type_e type){
    param_new_list_(&param_ctx,var, name, desc, type);
}
///==================================================================================

///=======================================BIN=======================================   
static void param_new_bin_(param_ctx_t * ctx,uint8_t * var, size_t len, const char * _name, const char * _desc, uint8_t def){
    param_t * f = param_new_param_(ctx, PARAM_BINARY, _name, _desc, true);
    f->ref = var;
    memset(var, def, len);
    f->list_bin_len = len;

}
void ParamBin  (uint8_t * var,size_t len, const char * name, const char * desc, uint8_t def_val){
    param_new_bin_(&param_ctx, var, len, name, desc, def_val);
}
///==================================================================================



static void *param_get_ref(param_t *param){
    return param->ref;
}

char *param_name(param_ctx_t * ctx, void *val){

    for(size_t i = 0; i < ctx->params_count; i++){
        param_t * f = &ctx->params[i];
        if(param_get_ref(f) == val){
            return (char*)f->name;
        }
    }
    return NULL;
}

char * ParamName(void *val){
    return param_name(&param_ctx, val);
}

bool ParseListParam(param_t * p, char * val_start_original, char * val_end_original){
    
    if (!p || !val_start_original || !val_end_original || *val_start_original != '[' || *val_end_original != ']') {
        return false;
    }

    size_t len = val_end_original - val_start_original - 1;
    char * input_copy = (char*)malloc(len + 1);
    if (!input_copy) {
        return false;
    }
    strncpy(input_copy, val_start_original + 1, len);
    input_copy[len] = '\0';

    char * current_element_start = input_copy;
    size_t idx = 0;

    char *saveptr;
    char *token = strtok_s(current_element_start, ", ", &saveptr);

    while(token != NULL) {
        if (strlen(token) == 0) {
            token = strtok_s(NULL, ", ", &saveptr);
            continue;
        }

        if (idx >= 32) {
            free(input_copy);
            return false; 
        }
        param_list_t * param_list = (param_list_t*)p->ref;

        switch (p->def.list.type) {
            case PARAM_BOOL:{
                if( strcmp(token, "1") == 0 ||  strcmp(token, "true") == 0  || strcmp(token, "yes") == 0 || strcmp(token, "si") == 0) {
                    param_list->items[idx].as_bool = true;
                } else if (strcmp(token, "0") == 0 ||  strcmp(token, "false") == 0  || strcmp(token, "no") == 0){
                    param_list->items[idx].as_bool = false;
                } else {
                    free(input_copy);
                    return false; // Valor booleano no válido
                }
            } break;
            
            case PARAM_UINT:{
                char * end;
                unsigned long int temp = strtoul(token, &end, 10);
                // Validar que toda la cadena fue consumida por strtoul
                if(*end != '\0' || temp > UINT_MAX){ // Comprobar también overflow si es necesario
                    free(input_copy);
                    return false;
                }
                param_list->items[idx].as_uint = (unsigned int)temp;
            } break;
            
            case PARAM_INT:{    
                char * end;
                long int temp = strtol(token, &end, 10);
                // Usar strtol y validar, es más seguro que atoi
                if(*end != '\0' || temp > INT_MAX || temp < INT_MIN){
                     free(input_copy);
                    return false;
                }
                param_list->items[idx].as_int = (int)temp;
            } break;
            
            case PARAM_FLOAT:{
                char * end;
                float temp = strtof(token, &end);
                 if(*end != '\0'){
                     free(input_copy);
                    return false;
                }
                param_list->items[idx].as_float = temp;
            } break;
            
            case PARAM_STR:{
                // CRUCIAL: Asumimos que p->val.list.items[idx]->as_str 
                // tiene suficiente espacio asignado (e.g., 256 bytes)
                strncpy(param_list->items[idx].as_str, token, sizeof(param_list->items[idx].as_str) - 1);
                param_list->items[idx].as_str[sizeof(param_list->items[idx].as_str) - 1] = '\0'; // Asegurar terminador nulo
            } break;
        
            default:
                free(input_copy);
                return false;
        }

        idx++;
        param_list->count = idx; // Mantener el contador actualizado
        token = strtok_s(NULL, ", ", &saveptr); // Obtener el siguiente token
    }

    free(input_copy); // Liberar la copia de la cadena
    p->has_changed = true;
    return true;
}


bool param_parse_txt(param_ctx_t * c, const char * filename){

    if(!strstr(filename, ".txt")){
        printf("file with erroneou sufix\n");
        return false;
    }

    // context_reset();

    FILE * fileh = fopen(filename, "r");
    if(!fileh){
        printf("file not found\n");
        return false;
    }

    char line[1024];

    bool mandatory_failed = false;
    bool found = false;
    while (fgets(line, sizeof line, fileh)) {
        

        if(*line == '#' || *line == ' ' || *line == '\n')continue;

        for (size_t i = 0; i < c->params_count; ++i) {
            char * name_start = strstr(line, c->params[i].name);
            if(name_start){
                char name[256];
                strcpy(name, name_start);
                char * name_end = strstr(name, "=");
                if(!name_end)continue;
                *name_end = '\0';
                int j = 0;
                while(name[j++] != '\0'){
                    if(name[j] == ' '){
                        name[j] = '\0';
                        break;
                    }
                }
                if(strcmp(name, c->params[i].name) != 0)continue;

                char * val_start = strstr(name_start, "=");
                
                val_start++;
                if(!val_start)break;
                found = true;

                bool value_found = true;
                while(*val_start == ' '){ // evitate spaces and check for missing value
                    if(*val_start == '\n'){
                        value_found = false;
                    }
                    val_start++;
                }
                if(!value_found)break; // continue in while(line)

                char * val_end = NULL;
                if(c->params[i].type == PARAM_LIST){
                    val_end = strstr(val_start,"]");
                    val_end++;
                }else if(c->params[i].type == PARAM_STR){
                    val_start++;
                    val_end = strstr(val_start,"\"");
                }else{
                    val_end = strstr(val_start," ");
                    if(!val_end){
                        val_end = strstr(val_start,"\n");
                    }
                }
                if(!val_end)break;
                *val_end = '\0'; // Cut out any comment or symbol that is after

                

                switch (c->params[i].type){
                    case PARAM_BOOL:{
                        if( strcmp(val_start, "1") == 0 ||  strcmp(val_start, "true") == 0  || strcmp(val_start, "yes") == 0 || strcmp(val_start, "si") == 0) {
                            *(bool*)c->params[i].ref = true;
                        }else if (strcmp(val_start, "0") == 0 ||  strcmp(val_start, "false") == 0  || strcmp(val_start, "no") == 0){
                            *(bool*)c->params[i].ref = false;
                        }else{ 
                            c->param_error = PARAM_ERROR_UNKNOWN;
                            return false;
                        }
                        c->params[i].has_changed = true;
                    }break;
                    case PARAM_INT:{
                        *(int*)c->params[i].ref = atoi(val_start);
                        c->params[i].has_changed = true;
                    }break;
                    case PARAM_UINT:{
                        char * end;
                        unsigned long int temp = strtoul(val_start,&end,10);
                        if(*end != '\0'){
                            c->param_error = PARAM_ERROR_INVALID_NUMBER;
                            return false;
                        }
                        *(unsigned long int*)c->params[i].ref = temp;
                        c->params[i].has_changed = true;
                    }break;
                    case PARAM_FLOAT:{
                        char * end;
                        float temp = strtof(val_start,&end);
                        if(*end != '\0'){
                            c->param_error = PARAM_ERROR_INVALID_NUMBER;
                            return false;
                        }
                        *(float*)c->params[i].ref = temp;
                        c->params[i].has_changed = true;
                    }break;
                    case PARAM_STR:{
                        strcpy((char*)c->params[i].ref, val_start);
                        c->params[i].has_changed = true;
                    }break;
                    case PARAM_LIST:{
                    
                        val_end--;
                        ParseListParam(&c->params[i], val_start, val_end);
                    }break;
                    case PARAM_BINARY:{
                        if(strstr(val_start,"0x")){
                            val_start += 2;
                        }
                        if((val_end - val_start)/2 > MAX_BIN){
                            c->param_error = PARAM_ERROR_BIN_OVERFLOW;
                            return false;
                        }
                        size_t idx = 0;
                        while(val_start != val_end){ // FFAABBCCDDFFEEBBCCDD
                            char temp[3];
                            char * end;
                            strncpy(temp,val_start,2);
                            *(((uint8_t*)(c->params[i].ref))+idx) = (uint8_t)strtoul(temp,&end,16);
                            val_start+=2;
                            idx++;
                        }

                        c->params[i].has_changed = true;
                    }break;
                    
                    default:{
                        printf("unrecheable\n");
                        return false;
                    }
                }
            }
        }
    } 

    if (!found) {
        printf("not found\n");
        c->param_error = PARAM_ERROR_UNKNOWN;
        return false;
    }
    for (size_t i = 0; i < c->params_count; ++i) {
        if(c->params[i].is_mandatory && !c->params[i].has_changed){
            c->param_error = PARAM_ERROR_NO_VALUE;
            ParamPrintError(stdout);
            mandatory_failed = true;
        }
    }
    if(mandatory_failed){
        return false;
    }
    return true;
}

bool InitCSV  (const char * dec_sep, const char * col_sep, csv_fileh_t * fileh){
    if(*dec_sep == *col_sep)return false;

    param_ctx.csv_fileh = fileh;

    param_ctx.csv_fileh->item_count = 0;
    param_ctx.csv_fileh->col_count = 0;

    param_ctx.csv_fileh->decimal_sep = (char*)dec_sep;
    param_ctx.csv_fileh->column_sep = (char*)col_sep;
    return true;
}

bool AddRowCSV(float * row, size_t row_n){
    if(row_n != param_ctx.csv_fileh->col_count)return false;

    int n = param_ctx.csv_fileh->item_count; 
    for(size_t i = 0; i < row_n; ++i){
        param_ctx.csv_fileh->csv_data[i].column_item[n] = row[i];
    }
    param_ctx.csv_fileh->item_count++;
    return true;
}

int GetCSVData(const char * name, float * data, size_t data_len){
    char low_name[256];
    strcpy(low_name, name);

    for (int i = 0; low_name[i] != '\0'; i++) {
        low_name[i] = (char)tolower((unsigned char)low_name[i]);
    }
    if(data_len < param_ctx.csv_fileh->item_count)return -1;
    size_t i;
    bool found = false;
    for(i = 0; i < param_ctx.csv_fileh->col_count; ++i){
        if(strstr(param_ctx.csv_fileh->csv_data[i].column_name, low_name)){
            found = true;
            break;
        }
    }
    if(!found)return false;

    for(size_t j = 0; j < param_ctx.csv_fileh->item_count; ++j){
        data[j] = param_ctx.csv_fileh->csv_data[i].column_item[j];
    }

    return param_ctx.csv_fileh->item_count;
}

void csv_new_headers(char * line, size_t size, param_ctx_t * c){
    char * token;
    char * saveptr;
    size_t idx = 0;

    if (size > 0 && line[size - 1] == '\n') {
        line[size - 1] = '\0';
    } else if (size > 0 && line[size - 1] == '\r') {
        line[size - 1] = '\0';
        if (size > 1 && line[size - 2] == '\n') {
             line[size - 2] = '\0';
        }
    }
    token = strtok_s(line, c->csv_fileh->column_sep, &saveptr);

    while (token != NULL) {
        for (int i = 0; token[i] != '\0'; i++) {
            token[i] = (char)tolower((unsigned char)token[i]);
        }
        if (idx < MAX_BIN) {
            strncpy(c->csv_fileh->csv_data[idx].column_name, token, sizeof(c->csv_fileh->csv_data[idx].column_name) - 1);
            c->csv_fileh->csv_data[idx].column_name[sizeof(c->csv_fileh->csv_data[idx].column_name) - 1] = '\0';
            
            char * space = strstr(c->csv_fileh->csv_data[idx].column_name, " ");
            char * scape = strstr(c->csv_fileh->csv_data[idx].column_name, "\n");
            if(space){
                *space = '\0';
            }else if(scape){
                *scape = '\0'; 
            }
            idx++;
        } else {
            fprintf(stderr, "Error: Demasiadas columnas en el CSV, superado el límite.\n");
            break;
        }
        token = strtok_s(NULL, c->csv_fileh->column_sep, &saveptr);
    }

    c->csv_fileh->col_count = idx;
}

bool param_parse_csv(param_ctx_t * c, const char * filename){
    if(!strstr(filename, ".csv"))return false;

    FILE * fileh = fopen(filename, "r");
    if(!fileh)return false;

    char line[2048];
    fgets(line, sizeof line, fileh);
    csv_new_headers(line, sizeof line, c);

    while (fgets(line, sizeof line, fileh)) {
        size_t len = strlen(line);
        if (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[len - 1] = '\0';
            if (len > 1 && line[len - 2] == '\r') {
                line[len - 2] = '\0';
            }
        }
        
        char *token;
        char *saveptr;
        size_t cols = 0;
        size_t row_index = 0;
        token = strtok_s(line, c->csv_fileh->column_sep, &saveptr);

        while (token != NULL) {

            if (cols >= MAX_BIN) {
                fprintf(stderr, "Error: Demasiadas columnas en la fila.\n");
                break;
            }

            size_t current_item_idx = c->csv_fileh->item_count;
            if (current_item_idx >= MAX_BIN) {
                 fprintf(stderr, "Error: Demasiados items en la columna %zu.\n", cols);
                 c->param_error = PARAM_ERROR_INVALID_NUMBER;
                 return false;
            }

            if (strlen(token) == 0) {
                c->csv_fileh->csv_data[cols].column_item[current_item_idx] = 0.0f;
            } else {
                if (*c->csv_fileh->decimal_sep != '.') {
                    char *dec_sep_pos = strstr(token, c->csv_fileh->decimal_sep);
                    if (dec_sep_pos != NULL) {
                        *dec_sep_pos = '.';
                    }
                }

                char *end;
                float temp = strtof(token, &end);

                if (*end != '\0') {
                    c->param_error = PARAM_ERROR_INVALID_NUMBER;
                    return false;
                }
                
                c->csv_fileh->csv_data[cols].column_item[current_item_idx] = temp;
            }

            cols++;
            token = strtok_s(NULL, c->csv_fileh->column_sep, &saveptr);
        }
        c->csv_fileh->item_count++;
        row_index++;
    }

    c->param_error = PARAM_NO_ERROR;
    return true;
}

static bool param_save_txt(param_ctx_t * c, const char * filename){
    if(c->params_count == 0){

        return false;
    }
    FILE * fouth = fopen(filename, "w");
    if(!fouth){
        fclose(fouth);
        return false;
    }


    for(size_t i = 0; i < c->params_count; ++i){
        switch (c->params[i].type){
            case PARAM_BOOL:{
                fprintf(fouth,"%s=%s #%s\n",
                    c->params[i].name, 
                    *(bool*)c->params[i].ref ? "true":"false",
                    c->params[i].desc);
            }break;
            case PARAM_UINT:{
                fprintf(fouth,"%s=%d #%s\n",
                    c->params[i].name, 
                    *(uint32_t*)c->params[i].ref,
                    c->params[i].desc);
            }break;
            case PARAM_INT:{
                fprintf(fouth,"%s=%i #%s\n",
                    c->params[i].name, 
                    *(int*)c->params[i].ref,
                    c->params[i].desc);
            }break;
            case PARAM_FLOAT:{
                fprintf(fouth,"%s=%.6f #%s\n",
                    c->params[i].name, 
                    *(float*)c->params[i].ref,
                    c->params[i].desc);
            }break;
            case PARAM_STR:{
                fprintf(fouth,"%s=\"%s\" #%s\n",
                    c->params[i].name, 
                    (char*)c->params[i].ref,
                    c->params[i].desc);
            }break;
            case PARAM_LIST:{
                fprintf(fouth,"%s=[", c->params[i].name);
                param_list_t * list = (param_list_t*)c->params[i].ref;
                for(size_t j = 0; j < list->count; ++j){
                    const char *sep = (j == (list->count - 1)) ? "]" : ",";
                    
                    switch (list->type){
                        case PARAM_BOOL:fprintf(fouth, "%s%s", list->items[j].as_bool ? "true":"false", sep);break;
                        case PARAM_UINT:fprintf(fouth, "%d%s", list->items[j].as_uint, sep);break;
                        case PARAM_INT: fprintf(fouth, "%i%s", list->items[j].as_int, sep);break;
                        case PARAM_FLOAT:fprintf(fouth, "%.6f%s", list->items[j].as_float, sep);break;
                        case PARAM_STR:fprintf(fouth, "\"%s\"%s", list->items[j].as_str, sep);break;
                        default:fclose(fouth);return false;
                    }
                }
                fprintf(fouth,"\n");

            }break;
            case PARAM_BINARY:{
                fprintf(fouth, "%s=0x",c->params[i].name);
                for(size_t j = 0; j < c->params[i].list_bin_len; ++j){
                    fprintf(fouth, "%02X", *( (uint8_t*)c->params[i].ref + j ) );
                }
                fprintf(fouth,"\n");
            }break;
            default:fclose(fouth);return false;
        }
    }
    fclose(fouth);
    return true;
}


static bool param_save_csv(param_ctx_t * c, const char * filename){
    if(c->csv_fileh->col_count == 0 || c->csv_fileh->item_count == 0)return false;

    FILE * fouth = fopen(filename, "w");
    if(!fouth)return false;


    // headers
    for(size_t i = 0; i < c->csv_fileh->col_count; ++i){
        if(i == (c->csv_fileh->col_count - 1))fprintf(fouth,"%s", c->csv_fileh->csv_data[i].column_name);
        else fprintf(fouth,"%s,", c->csv_fileh->csv_data[i].column_name);
    }
    fprintf(fouth,"\n");



    // data
    for(size_t j = 0; j < c->csv_fileh->item_count; ++j){
        for(size_t i = 0; i < c->csv_fileh->col_count; ++i){
            if(i == (c->csv_fileh->col_count - 1))fprintf(fouth,"%f", c->csv_fileh->csv_data[i].column_item[j]);
            else fprintf(fouth,"%f,", c->csv_fileh->csv_data[i].column_item[j]);
        }
        fprintf(fouth,"\n");
    }

    fclose(fouth);
    return true;
}

bool  ParamSave(const char * filename, file_type_e type){
    switch (type){
        case FILE_TYPE_TXT:return param_save_txt(&param_ctx, filename);
        case FILE_TYPE_CSV:return param_save_csv(&param_ctx, filename);
        default:return false;
    }
}

bool ParamParse(const char * filename, file_type_e type){
    switch (type){
        case FILE_TYPE_TXT:return param_parse_txt(&param_ctx, filename);
        case FILE_TYPE_CSV:return param_parse_csv(&param_ctx, filename);
        default:return false;
    }
}

void ParamPrintError(FILE * stream){
    param_ctx_t * fc = &param_ctx;
    switch(fc->param_error){
        case PARAM_NO_ERROR : {
            fprintf(stream,"Task failed succesfully, No error\n");
        break;
        }
        case PARAM_ERROR_UNKNOWN : {
            fprintf(stream,"ERROR: unknown param\n");
        break;
        }
        case PARAM_ERROR_NO_VALUE : {
            fprintf(stream,"ERROR: no value provided\n");
        break;
        }
        case PARAM_ERROR_INVALID_NUMBER : {
            fprintf(stream,"ERROR: invalid number\n");
        break;
        }
        case PARAM_ERROR_INVALID_FILE_EXT : {
            fprintf(stream,"ERROR: invalid file extension\n");
        break;
        }
        case PARAM_ERROR_INVALID_SIZE_SUFFIX : {
            fprintf(stream, "ERROR: invalid size suffix\n");
        break;
        }
        case PARAM_ERROR_BIN_OVERFLOW : {
            fprintf(stream, "ERROR: bin overflow\n");
        break;
        }
        default:
            assert(0 && "unreachable");
            exit(-1);
        break;
    } 
}
#endif // PARSER_IMP

#endif // FILE_PARSER_H_