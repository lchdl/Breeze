#include "basedefs.h"

bool chInStr(const char ch, const char* s) {
    for (int i = 0; s[i] != '\0'; i++) {
        if (ch == s[i]) return true;
    }
    return false;
}
void strSplit(char * src, char * buf, int bufLen, const char * delim, int* count)
{
    if (count) { /* count parts */
        *count = 1;
        int l = int(strlen(src));
        for (int i = 0; i < l; i++) {
            if (chInStr(src[i], delim))
                (*count)++;
        }
    }
    else { /* start split */
        if (src[0] == '\0') {
            buf[0] = '\0'; /* no more string to split, set buf="" and return */
            return;
        }
        size_t i = 0, s = 0, l = strlen(src);
        while (s < l) {
            int c = src[s++];
            if (chInStr(char(c), delim))
                break;
            if (int(i) < bufLen - 1) {
                buf[i] = char(c);
                i++;
            }
        }
        buf[i] = '\0'; /* reserve last character as '\0' */
        /* offset source string by s characters */
        int j = 0;
        for (; s + j < l; j++) {
            src[j] = src[s + j];
        }
        src[j] = '\0';
    }
}
void strRemCh(const char * src, char * buf, int bufLen, char ch)
{
    memset(buf, 0, bufLen);
    int _s = 0, _t = 0;
    while (_s < int(strlen(src))) {
        if (src[_s] == ch) {
            _s++;
            continue;
        }
        if (_t >= bufLen-1)
            break;
        buf[_t++] = src[_s++];
    }
}
bool txtGetWord(FILE * fp, char * buf, int bufLen, const char* wordDelim, const char commentChar)
{
    if (feof(fp)) return false;
    int i = 0;
    int write = 0, end = 0;
    while (!feof(fp) && !end) {
        int c = fgetc(fp);

        if (chInStr(char(c), wordDelim) || c == -1) {
            if (c == commentChar) { /* skip comment */
                while ((c = fgetc(fp)) != EOF)
                    if (c == '\n')
                        break;
            }
            else if (write == 1) { /* a word is finished because it meets a delimiter */
                write = 0;
                end = 1;
            }
        }
        else { /* word still not complete, write to buffer */
            write = 1;
        }

        if (write && i < bufLen - 1) {
            buf[i] = char(c);
            i++;
        }
    }
    buf[i] = '\0'; /* reserve last character as '\0' */
    if (i == 0) return false; /* if parsed word is empty string return false */
    else return true;
}
bool strGetWord(char * src, char * buf, int bufLen, const char * wordDelim, const char commentChar)
{
    if (!src) return false;
    int i = 0;
    size_t s = 0, l = strlen(src);
    int write = 0, end = 0;
    while (s < l && !end) {
        int c = src[s++];

        if (chInStr(char(c), wordDelim)) {
            if (c == commentChar) { /* skip comment */
                while ((c = src[s++]) && (s < l))
                    if (c == '\n')
                        break;
            }
            else if (write == 1) { /* a word is finished because it meets a delimiter */
                write = 0;
                end = 1;
            }
        }
        else { /* word still not complete, write to buffer */
            write = 1;
        }

        if (write && i < bufLen - 1) {
            buf[i] = char(c);
            i++;
        }
    }
    buf[i] = '\0'; /* reserve last character as '\0' */
    /* offset source string by s characters */
    int j = 0;
    for (; s + j < l; j++) {
        src[j] = src[s + j];
    }
    src[j] = '\0';
    /* if parsed word is empty string return false */
    if (i == 0) return false;
    else return true;
}
bool txtGetReal(FILE * fp, REAL * v)
{
    char* errptr = NULL, buf[32];
    if (!txtGetWord(fp, buf, 32, " #\n\r\t/", '#')) return false;
    *v = (REAL)(strtod(buf, &errptr));
    if (*errptr != '\0') return false;
    else return true;
}
bool txtGetInt(FILE * fp, int * v)
{
    char* errptr = NULL, buf[32];
    if (!txtGetWord(fp, buf, 32, " #\n\r\t/", '#')) return false;
    *v = (int)(strtol(buf, &errptr, 10));
    if (*errptr != '\0') return false;
    else return true;
}
bool txtGetVec3(FILE * fp, VEC3 * v)
{
    if (!txtGetReal(fp, &(v->e[X]))) return false;
    if (!txtGetReal(fp, &(v->e[Y]))) return false;
    if (!txtGetReal(fp, &(v->e[Z]))) return false;
    return true;
}
bool txtGetInt3(FILE * fp, INT3 * v)
{
    if (!txtGetInt(fp, &(v->e[X]))) return false;
    if (!txtGetInt(fp, &(v->e[Y]))) return false;
    if (!txtGetInt(fp, &(v->e[Z]))) return false;
    return true;
}

bool loadTextFile(char * file, char * buf, int bufLen) {
    FILE* fp = NULL;
    fp = fopen(file, "rb");
    if (fp == NULL) {
        return false;
    }
    else {
        /* get file size */
        fseek(fp, 0, SEEK_END);
        int flen = int(ftell(fp));
        fseek(fp, 0, SEEK_SET);
        /* load file into mem */
        int readlen = flen > (bufLen - 1) ? (bufLen - 1) : flen;
        bool success = true;
        if (buf) {
            memset(buf, 0, bufLen);
            int len = int(fread(buf, 1, readlen, fp));
            if (len != readlen) { success = false; }
            else { success = true; }
            /* add '\0' to the end of the string */
            buf[readlen + 1] = '\0';
        }
        else { success = false; }
        fclose(fp);
        return success;
    }
}
Array<BYTE> loadAsByteArray(const char * file)
{
    Array<BYTE> byteArray;
    FILE* fp;
    if ((fp = fopen(file, "rb")) == NULL)
        return byteArray;
    BYTE byte;
    while (true) {
        if (fread(&byte, 1, 1, fp) != 1)
            break;
        else
            byteArray.append(byte);
    }
    fclose(fp);
    return byteArray;
}
bool saveByteArray(const Array<BYTE>& byteArray, const char * file)
{
    FILE* fp = NULL;
    if ((fp = fopen(file, "wb")) == NULL)
        return false;
    for (int i = 0; i < byteArray.size(); i++) {
        if (fwrite(&(byteArray[i]), 1, 1, fp) != 1){
            fclose(fp);
            return false;
        }
    }
    fclose(fp);
    return true;
}
void printVec3(const char* prefix, VEC3 v)
{
    if (prefix) {
        printf("%s(%.2f, %.2f, %.2f)\n", prefix ,v.x, v.y, v.z);
    }
    else {
        printf("(%.2f, %.2f, %.2f)\n", v.x, v.y, v.z);
    }
}
void printQuaternion(const char * prefix, QUATERNION q)
{
    if (prefix != NULL) {
        printf("%s", prefix);
    }
    printf("%.2fi + %.2fj + %.2fk + %.2fs\n", q.x, q.y, q.z, q.s);
}
void printByteArray(const Array<BYTE>& byteArray)
{
    for (int i = 0; i < byteArray.size(); i++)
        printf("%c", char(byteArray[i]));
}
void printMat3x3(const char * prefix, MAT3x3 m)
{
    if (prefix != NULL) {
        printf("%s\n", prefix);
    }
    printf("%.2f %.2f %.2f\n", m.xx, m.xy, m.xz);
    printf("%.2f %.2f %.2f\n", m.yx, m.yy, m.yz);
    printf("%.2f %.2f %.2f\n", m.zx, m.zy, m.zz);
}

