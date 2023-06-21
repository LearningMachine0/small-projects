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

unsigned long M_last_iter;
mpz_t M_last_num;
mpz_t M_last_den;
static inline void M(mpz_t result, unsigned long q)
{
    // numerator product n=0 -> q-1: ((12n+6)^3 - (192n + 96))
    // denominator product n=0 -> q-1: (n+1)^3
    mpz_t a, b, c;
    mpz_inits(a, b, c, NULL);
    mpz_t prod_n, prod_d, prod;
    mpz_inits(prod_n, prod_d, prod, NULL);
    unsigned long last_iter;
    #pragma omp critical
    {
        mpz_set(prod_n, M_last_num);
        mpz_set(prod_d, M_last_den);
        // Copy last iteration number to prevent race condition
        last_iter = M_last_iter;
    }

    // Calculate M based on previous value of M
    if (q > last_iter)
    {
        for (unsigned long n = last_iter; n < q; n++)
        {
            mpz_set_ui(a, n);
            mpz_mul_ui(a, a, 12);
            mpz_add_ui(a, a, 6);
            mpz_pow_ui(a, a, 3);

            mpz_set_ui(b, n);
            mpz_mul_ui(b, b, 192);
            mpz_add_ui(b, b, 96);

            mpz_sub(a, a, b);
            mpz_mul(prod_n, prod_n, a);

            mpz_set_ui(c, n);
            mpz_add_ui(c, c, 1);
            mpz_pow_ui(c, c, 3);
            mpz_mul(prod_d, prod_d, c);
        }
    }
    else if (q < last_iter)
    {
        for (unsigned long n = last_iter - 1; n >= q; n--)
        {
            mpz_set_ui(a, n);
            mpz_mul_ui(a, a, 12);
            mpz_add_ui(a, a, 6);
            mpz_pow_ui(a, a, 3);

            mpz_set_ui(b, n);
            mpz_mul_ui(b, b, 192);
            mpz_add_ui(b, b, 96);

            mpz_sub(a, a, b);
            mpz_divexact(prod_n, prod_n, a);
            
            mpz_set_ui(c, n);
            mpz_add_ui(c, c, 1);
            mpz_pow_ui(c, c, 3);
            mpz_divexact(prod_d, prod_d, c);
        }
    }
    else if (q == last_iter)
    {
        //#pragma omp critical
        mpz_set(prod_n, M_last_num);
        mpz_set(prod_d, M_last_den);
    }
    mpz_clears(a, b, c,  NULL);
    mpz_set(prod, prod_n);
    mpz_divexact(prod, prod, prod_d);
    mpz_set(result, prod);
    
    // Prevent race condition when writing to global variable
    #pragma omp critical
    if (q % 5 == 0)
    {
        M_last_iter = q;
        mpz_set(M_last_num, prod_n);
        mpz_set(M_last_den, prod_d);
    }
    mpz_clears(prod, prod_n, prod_d, NULL);
    return;
}

void pi(mpf_t result, unsigned long prec_bits, unsigned long prec_digits)
{
    // intialisation for M function
    mpz_inits(M_last_num, M_last_den, NULL);
    M_last_iter = 0;
    mpz_set_ui(M_last_num, 1);
    mpz_set_ui(M_last_den, 1);

    // constant
    mpf_t constant, final;
    mpf_inits(constant, final, NULL);
    mpf_sqrt_ui(constant, 10005);
    mpf_mul_ui(constant, constant, 426880);

    unsigned long iters = prec_digits / 8;
    mpf_t sum;
    mpf_init(sum);

    #pragma omp parallel for schedule(dynamic)
    for(unsigned long k = 0; k < iters; k++)
    {
        mpz_t l, x, m;
        mpz_inits(l, x, m, NULL);
        L(l, k);
        X(x, k);
        M(m, k);

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

        mpz_clears(l, x, m, NULL);
        mpf_clears(n, d, NULL);
    }
    #pragma omp barrier

    mpf_div(final, constant, sum);
    mpf_set(result, final);
    mpf_clears(constant, final, sum, NULL);
    return;
}

int main(int argc, char* argv[])
{
    unsigned long prec_digits = atoi(argv[1]);
    unsigned long prec_bits = (prec_digits + 1) * BITS_PER_DIGIT + 3;
    mpf_set_default_prec(prec_bits);
    mpf_t r;
    mpf_init(r);

    pi(r, prec_bits, prec_digits);
    gmp_printf("%.*Ff\n", prec_digits, r);
    mpf_clear(r);
    return 0;
}
