/* Same double loop as toymodel.pair_forces_loop; compile_c.py builds it standalone and as a -DNO_MAIN ctypes library */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* pos and F are (n, 2) arrays stored row by row: x0 y0 x1 y1 ... */
void pair_forces_c(int n, const double *pos, double k, double r0, double rc, double *F)
{
    double rc2 = rc * rc;
    for (int i = 0; i < 2 * n; i++)
        F[i] = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = pos[2 * i] - pos[2 * j];       /* d[i, j] = r_i - r_j */
            double dy = pos[2 * i + 1] - pos[2 * j + 1];
            double r2 = dx * dx + dy * dy;
            if (r2 > 0.0 && r2 <= rc2) {               /* 0 < r <= rc, sqrt only for neighbours */
                double r = sqrt(r2);
                double fmag = -k * (r - r0);
                double fx = fmag * dx / r;
                double fy = fmag * dy / r;
                F[2 * i] += fx;
                F[2 * i + 1] += fy;
                F[2 * j] -= fx;
                F[2 * j + 1] -= fy;
            }
        }
    }
}

#ifndef NO_MAIN
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static int cmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    int n = 1000;
    if (argc > 1)
        n = atoi(argv[1]);
    double k = 15.0, r0 = 0.8, rc = 2.0;
    double L = sqrt(n / 0.12);    /* same density as the handout: 12 particles in 10 x 10 */
    int repeats = 7;

    double *pos = malloc(2 * n * sizeof(double));
    double *F = malloc(2 * n * sizeof(double));
    srand(0);
    for (int i = 0; i < 2 * n; i++)
        pos[i] = L * rand() / (double)RAND_MAX;

    /* tiny N takes microseconds, so each repeat runs the kernel several times */
    int inner = 1;
    while (inner < 1000000) {
        double t0 = now();
        for (int r = 0; r < inner; r++)
            pair_forces_c(n, pos, k, r0, rc, F);
        if (now() - t0 > 0.02)
            break;
        inner *= 2;
    }

    double t[16];
    for (int rep = 0; rep < repeats; rep++) {
        double t0 = now();
        for (int r = 0; r < inner; r++)
            pair_forces_c(n, pos, k, r0, rc, F);
        t[rep] = (now() - t0) / inner;
    }
    qsort(t, repeats, sizeof(double), cmp);

    double fsum_x = 0.0, fsum_y = 0.0;
    for (int i = 0; i < n; i++) {
        fsum_x += F[2 * i];
        fsum_y += F[2 * i + 1];
    }
    printf("N=%d L=%.4f repeats=%d inner=%d median=%.6e s min=%.6e s max=%.6e s |sumF|=%.3e\n",
           n, L, repeats, inner, t[repeats / 2], t[0], t[repeats - 1],
           sqrt(fsum_x * fsum_x + fsum_y * fsum_y));

    free(pos);
    free(F);
    return 0;
}
#endif
