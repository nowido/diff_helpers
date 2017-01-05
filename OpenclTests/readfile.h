#ifndef READFILE_H
#define READFILE_H

#include <stdio.h>
#include <stdlib.h>

char* readFile(const char* fileName)
{
    char* content = NULL;
    
    FILE* f = fopen(fileName, "rb");    

    if(f)
    {
        fseek(f, 0, SEEK_END);

        long size = ftell(f);                

        if(size > 0)
        {            
            fseek(f, 0, SEEK_SET);

            size_t stringLength = size + 1;

            content = (char*)malloc(stringLength);
            fread(content, size, 1, f);
            content[size] = 0;            
        }

        fclose(f);
    }

    return content;    
}

#endif