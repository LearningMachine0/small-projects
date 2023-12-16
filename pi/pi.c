/* Pi Calculator using the binary splitting method for the Chudnovsky Algorithm
 * from https://en.wikipedia.org/wiki/Chudnovsky_algorithm
*/

#include <omp.h>
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>

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

    mpz_t int_den;
    mpz_init(int_den);
    mpf_t den, f_Qab;
    mpf_inits(den, f_Qab, NULL);
    
    mpz_t Pab, Qab, Rab;
    mpz_inits(Pab, Qab, Rab, NULL);
    printf("binary split\n");
    binary_split(Pab, Qab, Rab, 1, n + 1);
    mpf_set_z(f_Qab, Qab);
    
    printf("calculating numerator\n");
    mpf_set_d(r_pi, 10005.0);
    mpf_sqrt(r_pi, r_pi);
    mpf_mul_ui(r_pi, r_pi, 426880);
    mpf_mul(r_pi, r_pi, f_Qab);
    
    printf("calculating denominator\n");
    mpz_mul_ui(int_den, Qab, 13591409);
    mpz_add(int_den, int_den, Rab);
    mpf_set_z(den, int_den);
    
    mpf_div(r_pi, r_pi, den);
    mpz_clears(int_den, Pab, Qab, Rab, NULL);
    mpf_clears(den, f_Qab, NULL);
    return;
}

int main(int argc, char* argv[])
{
    printf("getting prec\n");
    unsigned long prec = atol(argv[1]);
    unsigned long prec_bits = (prec + 2) * BITS_PER_DIGIT + 3;
    printf("prec: %ld\n", prec);
    printf("prec_bits: %ld\n", prec_bits);
    mpf_set_default_prec(prec_bits);
    mpf_t pi;
    mpf_init(pi);
    int n;
    if (prec / 14 > 1)
    {
        n = prec / 14 + 1;
    }
    else
    {
        n = 2;
    }
    chudnovsky(pi, prec / 14 + 1, 1, prec_bits);
    gmp_printf("%.*Ff\n", prec, pi);
    
    return 0;
}
