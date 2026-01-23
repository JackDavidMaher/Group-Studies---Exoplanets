#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
int main(void)
{
<<<<<<< HEAD
<<<<<<< HEAD
    // Directory containing spectrum images
=======
>>>>>>> 4d54eaf (Jack maher (#40))
=======
    // Directory containing spectrum images
>>>>>>> 1d70245 (Jack maher (#44))
    const char *dirpath = "SpectrumPlots";
    DIR *d = opendir(dirpath);
    if (!d)
    {
        fprintf(stderr, "opendir(%s) failed: %s\n", dirpath, strerror(errno));
        return 1;
    }

    struct dirent *ent;
    char fullpath[PATH_MAX];
    int errors = 0;
    while ((ent = readdir(d)) != NULL)
    {
<<<<<<< HEAD
<<<<<<< HEAD
        // skip current and parent directories
=======
        /* skip current/parent directories */
>>>>>>> 4d54eaf (Jack maher (#40))
=======
        // skip current and parent directories
>>>>>>> 1d70245 (Jack maher (#44))
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            continue;

        if (snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, ent->d_name) >= (int)sizeof(fullpath))
        {
            fprintf(stderr, "path too long: %s/%s\n", dirpath, ent->d_name);
            errors = 1;
            continue;
        }

        struct stat st;
        if (lstat(fullpath, &st) != 0)
        {
            fprintf(stderr, "stat(%s) failed: %s\n", fullpath, strerror(errno));
            errors = 1;
            continue;
        }

<<<<<<< HEAD
<<<<<<< HEAD
        // skips deleting directories (allows for storing of permentant files if needed)
=======
        /* skip directories; remove everything else (regular files, symlinks, etc.) */
>>>>>>> 4d54eaf (Jack maher (#40))
=======
        // skips deleting directories (allows for storing of permentant files if needed)
>>>>>>> 1d70245 (Jack maher (#44))
        if (S_ISDIR(st.st_mode))
        {
            fprintf(stderr, "skipping directory: %s\n", fullpath);
            continue;
        }

        if (remove(fullpath) != 0)
        {
            fprintf(stderr, "remove(%s) failed: %s\n", fullpath, strerror(errno));
            errors = 1;
            continue;
        }

        printf("deleted: %s\n", fullpath);
    }

    closedir(d);
    return errors ? 2 : 0;
}