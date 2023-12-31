/* Pi Calculator using the binary splitting method for the Chudnovsky Algorithm
 * from https://en.wikipedia.org/wiki/Chudnovsky_algorithm
*/

#include <omp.h>
#include <gmp.h>
#include <stdlib.h>
#include <stdio.h>

#define BITS_PER_DIGIT 3.32192809489

void binary_split(mpz_t r_Pab, mpz_t r_Qab, mpz_t r_Rab, long a, long b)
{
    // P(a, a+1) = -(6a-1)(2a-1)(6a-5)
    // Q(a, a+1) = 10939058860032000a**3
    // R(a, a+1) = P(a, a+1) * (545140134a + 13591409)
    mpz_t Pab, Qab, Rab;
    mpz_inits(Pab, Qab, Rab, NULL);
    // printf("binary split inside a=%ld b=%ld\n", a, b);
    
    if (b == a + 1)
    {
        // calculate P. initialize 3 variables for the terms
        mpz_t P1, P2, P3;
        
        mpz_init_set_si(P1, a);
        mpz_mul_ui(P1, P1, 6);
        mpz_sub_ui(P1, P1, 1);
        mpz_neg(P1, P1);
        
        mpz_init_set_si(P2, a);
        mpz_mul_ui(P2, P2, 2);
        mpz_sub_ui(P2, P2, 1);
        
        mpz_init_set_si(P3, a);
        mpz_mul_ui(P3, P3, 6);
        mpz_sub_ui(P3, P3, 5);
        
        //mpz_set(Pab, P1);
        mpz_mul(Pab, P1, P2);
        mpz_mul(Pab, Pab, P3);
        
        // calculate Q
        mpz_set_si(Qab, a);
        mpz_pow_ui(Qab, Qab, 3);
        mpz_mul_ui(Qab, Qab, 10939058860032000);
        
        // calculate R
        mpz_set_si(Rab, a);
        mpz_mul_ui(Rab, Rab, 545140134);
        mpz_add_ui(Rab, Rab, 13591409);
        mpz_mul(Rab, Rab, Pab);
        
        mpz_clears(P1, P2, P3, NULL);
    }
    else
    {
        long m = (a + b) / 2;
        mpz_t Pam, Qam, Ram, Pmb, Qmb, Rmb, Pam_Rmb;
        mpz_inits(Pam, Qam, Ram, Pmb, Qmb, Rmb, Pam_Rmb, NULL);
        
        binary_split(Pam, Qam, Ram, a, m);
        binary_split(Pmb, Qmb, Rmb, m, b);
        
        mpz_mul(Pab, Pam, Pmb);
        mpz_mul(Qab, Qam, Qmb);
        mpz_mul(Rab, Qmb, Ram);
        // use Pam_Rmb to calculate second term Pam * Rmb
        mpz_mul(Pam_Rmb, Pam, Rmb);
        mpz_add(Rab, Rab, Pam_Rmb);
        
        mpz_clears(Pam, Qam, Ram, Pmb, Qmb, Rmb, Pam_Rmb, NULL);
    }

    mpz_set(r_Pab, Pab);
    mpz_set(r_Qab, Qab);
    mpz_set(r_Rab, Rab);
    mpz_clears(Pab, Qab, Rab, NULL);
    return;
}

void chudnovsky(mpf_t r_pi, long n, int threads, unsigned long prec_bits)
{
    // (426880 * sqrt(10005) * Q(1, n)) / (13591409 * Q(1, n) + R(1, n))
    // to retain accuracy, denominator is initially an integer
    mpf_set_default_prec(prec_bits);

    mpz_t P[threads];
    mpz_t Q[threads];
    mpz_t R[threads];
    for (int i = 0; i < threads; i++)
    {
        mpz_inits(P[i], Q[i], R[i], NULL);
    }

    mpz_t int_den;
    mpz_init(int_den);
    mpf_t den, f_Qab;
    mpf_inits(den, f_Qab, NULL);
    
    // multiprocessing implementation
    long interval = n / threads;
    printf("n %ld\nthreads %d\n", n, threads);
    printf("interval %ld\n", interval);
    
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < threads - 1; i++)
    {
        long start = i * interval + 1;
        long end = (i + 1) * interval + 1;
        // printf("%ld %ld\n", start, end);
        binary_split(P[i], Q[i], R[i], start, end);
    }
    // printf("%ld %ld\n", (threads - 1) * interval + 1, n);
    binary_split(P[threads - 1], Q[threads - 1], R[threads - 1], (threads - 1) * interval + 1, n);
    #pragma omp barrier
    
    if (threads > 1)
    {
        for (int i = 1; i < threads; i++)
        {
            // printf("i %d\n", i);
            mpz_t temp_P, temp_Q, temp_R, P_R;
            mpz_inits(temp_P, temp_Q, temp_R, P_R, NULL);
            mpz_mul(temp_P, P[0], P[i]);
            mpz_mul(temp_Q, Q[0], Q[i]);
            mpz_mul(temp_R, Q[i], R[0]);
            mpz_mul(P_R, P[0], R[i]);
            mpz_add(temp_R, temp_R, P_R);

            mpz_set(P[0], temp_P);
            mpz_set(Q[0], temp_Q);
            mpz_set(R[0], temp_R);
            mpz_clears(P[i], Q[i], R[i], NULL);
            mpz_clears(temp_P, temp_Q, temp_R, NULL);
        }
    }

    mpf_set_z(f_Qab, Q[0]);
    
    printf("calculating numerator\n");
    mpf_set_d(r_pi, 10005.0);
    mpf_sqrt(r_pi, r_pi);
    mpf_mul_ui(r_pi, r_pi, 426880);
    mpf_mul(r_pi, r_pi, f_Qab);
    
    printf("calculating denominator\n");
    mpz_mul_ui(int_den, Q[0], 13591409);
    mpz_add(int_den, int_den, R[0]);
    mpf_set_z(den, int_den);
    
    mpf_div(r_pi, r_pi, den);

    mpz_clears(P[0], Q[0], R[0], NULL);
    mpf_clears(den, f_Qab, NULL);
    return;
}

int main(int argc, char* argv[])
{
    printf("getting prec\n");
    unsigned long prec = atol(argv[1]);
    // unsigned long prec = 200;
    unsigned long prec_bits = (prec + 2) * BITS_PER_DIGIT + 3;
    printf("prec: %ld\n", prec);
    printf("prec_bits: %ld\n", prec_bits);
    mpf_set_default_prec(prec_bits);
    mpf_t pi;
    mpf_init(pi);
    int threads = omp_get_max_threads();
    int n = (prec / 14 > 1) ? prec / 14 + 1 : 2;
    if (n < threads)
    {
        threads = n / 2;
    }
    chudnovsky(pi, n, threads, prec_bits);
    gmp_printf("%.*Ff\n", prec, pi);
    
    return 0;
}
