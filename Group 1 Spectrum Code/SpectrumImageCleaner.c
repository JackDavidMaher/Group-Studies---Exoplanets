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
    const char *dirpath = "..";
    const char prefix[] = "TransmissionSpectrum";
=======
    // Directory containing spectrum images
    const char *dirpath = "SpectrumPlots";
>>>>>>> 1d70245 (Jack maher (#44))
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
        if (strncmp(ent->d_name, prefix, sizeof(prefix) - 1) != 0)
=======
        // skip current and parent directories
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
>>>>>>> 1d70245 (Jack maher (#44))
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