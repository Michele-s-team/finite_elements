tol = 1e-4
L = 2.2
h = 0.41

system("sed -i '' 's/,/ /g' ~/Desktop/output.csv ")
system("awk '/0 2.2 /' ~/Desktop/output.csv > ~/Desktop/output_L.csv")

#check(x) =  (x > L/4 && x < 3*L/4) ? 1 : 0
#check(x) =  (abs(x-L) < tol) ? 1 : 0
#check(x) =  (x > 7*L/8 < tol) ? 1 : 0
#f(x) = - 0.5 * ((1.0 - 0.0)/L) * x*(x-h)
A = 0.9836250000000001
f(x) = A * x*(h-x)/(h**2)

p '/Users/michele/Desktop/output_L.csv' u 5:1, f(x)
