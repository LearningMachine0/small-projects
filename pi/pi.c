/* Pi calculator using the Chudnovsky algorithm
 * Written by: Kai-Yu She (June 2023)
 * Formulas from https://en.wikipedia.org/wiki/Chudnovsky_algorithm
 * To compile: gcc pi.c -o pi -fopenmp -lgmp
 * - Required to have GNU GMP installed on system
*/

#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>

#define BITS_PER_DIGIT 3.32192809489
omp_lock_t rw_M;

static inline void L(mpz_t result, unsigned long q)
{
    // 545140134q + 13591409
    mpz_set_ui(result, 545140134);
    mpz_mul_ui(result, result, q);
    mpz_add_ui(result, result, 13591409);
    return;
}

static inline void X(mpz_t result, unsigned long q)
{
    // (-262537412640768000)^q
    mpz_set_ui(result, 262537412640768000);
    mpz_pow_ui(result, result, q);
    if (q % 2 != 0)
    {
        mpz_mul_si(result, result, -1);
    }
    return;
}

mpz_t last_M_num;
mpz_t last_M_den;
unsigned int last_M_q;
void M(mpz_t rv, mpz_t n, mpz_t d, unsigned long q)
{
    // ((12q + 6)^3 - (192q + 96))/(q+1)^3
    mpz_t a, b, c, result, num, den;
    mpz_inits(a, b, c, result, num, den, NULL);
    unsigned int last_q;

    omp_set_lock(&rw_M);
    last_q = last_M_q;
    mpz_set(num, last_M_num);
    mpz_set(den, last_M_den);
    omp_unset_lock(&rw_M);

    // Calculate M based on previous value of M
    if (q > last_q)
    {
        for (unsigned long n = last_q; n < q; n++)
        {
            mpz_set_ui(a, n);
            mpz_mul_ui(a, a, 12);
            mpz_add_ui(a, a, 6);
            mpz_pow_ui(a, a, 3);

            mpz_set_ui(b, n);
            mpz_mul_ui(b, b, 192);
            mpz_add_ui(b, b, 96);

            mpz_sub(a, a, b);
            mpz_mul(num, num, a);

            mpz_set_ui(c, n);
            mpz_add_ui(c, c, 1);
            mpz_pow_ui(c, c, 3);
            mpz_mul(den, den, c);
        }
    }
    else if (q < last_q)
    {
        for (unsigned long n = last_q - 1; n >= q; n--)
        {
            mpz_set_ui(a, n);
            mpz_mul_ui(a, a, 12);
            mpz_add_ui(a, a, 6);
            mpz_pow_ui(a, a, 3);

            mpz_set_ui(b, n);
            mpz_mul_ui(b, b, 192);
            mpz_add_ui(b, b, 96);

            mpz_sub(a, a, b);
            mpz_divexact(num, num, a);

            mpz_set_ui(c, n);
            mpz_add_ui(c, c, 1);
            mpz_pow_ui(c, c, 3);
            mpz_divexact(den, den, c);
        }
    }
    mpz_divexact(result, num, den);

    mpz_set(rv, result);
    mpz_set(n, num);
    mpz_set(d, den);
    mpz_clears(a, b, c, result, num, den, NULL);
    return;
}

void pi(mpf_t result, unsigned long prec_bits, unsigned long prec_digits)
{
    // intialisation for M function
    last_M_q = 0;
    mpz_inits(last_M_num, last_M_den, NULL);
    mpz_set_ui(last_M_num, 1);
    mpz_set_ui(last_M_den, 1);
    int M_change_interval = 10; // for last_M_* variables

    // omp_init_lock(&rw_M);

    // constant
    mpf_t constant, final;
    mpf_inits(constant, final, NULL);
    mpf_sqrt_ui(constant, 10005);
    mpf_mul_ui(constant, constant, 426880);

    // round up division
    unsigned long iters = (prec_digits + 8 - 1) / 8;
    mpf_t sum;
    mpf_init(sum);

    #pragma omp parallel for schedule(dynamic) shared(sum, last_M_q, last_M_num, last_M_den)
    for(unsigned long k = 0; k < iters; k++)
    {
        mpz_t l, x, m, M_num, M_den;
        mpz_inits(l, x, m, M_num, M_den, NULL);
        // calculate iteration
        L(l, k);
        X(x, k);
        M(m, M_num, M_den, k);

        // calculate numerator and denominator
        mpf_t n, d; 
        mpf_inits(n, d, NULL);
        mpz_mul(m, m, l); // reusing `m' to calculate numerator
        mpf_set_z(n, m);
        mpf_set_z(d, x);
        mpf_div(n, n, d);

       #pragma omp critical
        {
            mpf_add(sum, sum, n);
        }

        if ((k % M_change_interval) == 0 && k > last_M_q)
        {
            omp_set_lock(&rw_M);
            // #pragma omp critical
            // {
                last_M_q = k;
                mpz_set(last_M_num, M_num);
                mpz_set(last_M_den, M_den);
            // }
            omp_unset_lock(&rw_M);
        }

        mpz_clears(l, x, m, M_num, M_den, NULL);
        mpf_clears(n, d, NULL);
    }
    #pragma omp barrier

    mpf_div(final, constant, sum);
    mpf_set(result, final);
    mpf_clears(constant, final, sum, NULL);
    mpz_clears(last_M_num, last_M_den, NULL);
    omp_destroy_lock(&rw_M);
    return;
}

int main(int argc, char* argv[])
{
    unsigned long prec_digits = atoi(argv[1]);
    unsigned long prec_bits = (prec_digits + 1) * BITS_PER_DIGIT + 3;
    mpf_set_default_prec(prec_bits);
    mpf_t r;
    mpf_init(r);
    omp_init_lock(&rw_M);

    pi(r, prec_bits, prec_digits);
    gmp_printf("%.*Ff\n", prec_digits, r);
    mpf_clear(r);
    return 0;
}
