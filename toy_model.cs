using System;

namespace CSharp
{
    public static class ToyModel
    {
        static Random rng = new Random();

        public static double MC(Func<double[], double> minus_prob, double[] exps, double[,] m, int N)
        {
            double psum = 0.0;
        
            int n_exps = exps.Length;
            int n_px = m.GetLength(0);
            
            double[] s1 = new double[2];
            
            for(int k=0; k<N; k++)
            {
                if(k % 1000 == 0)
                    System.Console.WriteLine(k.ToString());
                
                s1[0] = 6.0 + 0.5 * rng.NextDouble();
                s1[1] = -2.0 + 3.0 * rng.NextDouble();

                double p = minus_prob(s1);
                psum += p;
                for(int i=0; i<n_exps; i++)
                {
                    double fv = s1[0] + s1[1] * exps[i];
                    int j = (int) ((fv - 1.5) / (6.0 - 1.5) * n_px);
                    if(0 <= j && j < n_px)
                        m[j, i] += p;
                }
            }
            
            return psum;
        }
    }
}