import sympy as sp
from sympy.printing.numpy import NumPyPrinter

t = sp.symbols("t")
l1 = sp.symbols("l1")
m1 = sp.symbols("m1")
l2 = sp.symbols("l2")
m2 = sp.symbols("m2")
l3 = sp.symbols("l3")
m3 = sp.symbols("m3")
l4 = sp.symbols("l4")
m4 = sp.symbols("m4")

g = sp.symbols("g")

th1 = sp.Function("th1")(t)
th2 = sp.Function("th2")(t)
th3 = sp.Function("th3")(t)
th4 = sp.Function("th4")(t)

x1 = -l1 * sp.sin(th1) 
y1 =  l1 * sp.cos(th1) 

x2 = -l1 * sp.sin(th1) - l2 * sp.sin(th2) 
y2 =  l1 * sp.cos(th1) + l2 * sp.cos(th2) 

x3 = -l1 * sp.sin(th1) - l2 * sp.sin(th2) - l3 * sp.sin(th3) 
y3 =  l1 * sp.cos(th1) + l2 * sp.cos(th2) + l3 * sp.cos(th3)

x4 = -l1 * sp.sin(th1) - l2 * sp.sin(th2) - l3 * sp.sin(th3) - l4 * sp.sin(th4)
y4 =  l1 * sp.cos(th1) + l2 * sp.cos(th2) + l3 * sp.cos(th3) + l4 * sp.cos(th4)

dx1 = sp.diff(x1, t)
dy1 = sp.diff(y1, t)

dx2 = sp.diff(x2, t)
dy2 = sp.diff(y2, t)

dx3 = sp.diff(x3, t)
dy3 = sp.diff(y3, t)

dx4 = sp.diff(x4, t)
dy4 = sp.diff(y4, t)

dth1 = sp.diff(th1, t)
dth2 = sp.diff(th2, t)
dth3 = sp.diff(th3, t)
dth4 = sp.diff(th4, t)

print(" ------------------------------------------------")
print("               dth1 dth2 dth3 dth4 expression              ")
print(" ------------------------------------------------")
sp.pprint(dth1)
sp.pprint(dth2)
sp.pprint(dth3)
sp.pprint(dth4)

print(" ------------------------------------------------")
print("               dx1 expression                    ")
print(" ------------------------------------------------")
sp.pprint(dx1)

print(" ------------------------------------------------")
print("               dx2 expression                    ")
print(" ------------------------------------------------")
sp.pprint(dx2)

print(" ------------------------------------------------")
print("               dx3 expression                    ")
print(" ------------------------------------------------")
sp.pprint(dx3)

print(" ------------------------------------------------")
print("               dx4 expression                    ")
print(" ------------------------------------------------")
sp.pprint(dx4)

print(" ------------------------------------------------")
print("               dy1 expression                    ")
print(" ------------------------------------------------")
sp.pprint(dy1)

print(" ------------------------------------------------")
print("               dy2 expression                    ")
print(" ------------------------------------------------")
sp.pprint(dy2)

print(" ------------------------------------------------")
print("               dy3 expression                    ")
print(" ------------------------------------------------")
sp.pprint(dy3)

print(" ------------------------------------------------")
print("               dy4 expression                    ")
print(" ------------------------------------------------")
sp.pprint(dy4)

print(" ------------------------------------------------")
print("               T expression                    ")
print(" ------------------------------------------------")
T = 1/2 * m1 * (dx1**2 + dy1**2)  + 1/2 * m2 * (dx2**2 + dy2**2) + 1/2 * m3 * (dx3**2 + dy3**2) + 1/2 * m4 * (dx4**2 + dy4**2)
sp.pprint(T)

print(" ------------------------------------------------")
print("               V expression                    ")
print(" ------------------------------------------------")
V = m1*g*y1 + m2*g*y2 + m3*g*y3 + m4*g*y4
sp.pprint(V)

print(" ------------------------------------------------")
print("               L expression                  ")
print(" ------------------------------------------------")
L = T - V
sp.pprint(L)

print(" ------------------------------------------------")
print("               dynamic expression 1              ")
print(" ------------------------------------------------")
eq1 = sp.diff( sp.diff(L, dth1), t) - sp.diff(L, th1)
eq1 = sp.trigsimp(eq1)
sp.pprint(eq1)

print(" ------------------------------------------------")
print("               dynamic expression 2              ")
print(" ------------------------------------------------")
eq2 = sp.diff( sp.diff(L, dth2), t) - sp.diff(L, th2)
eq2 = sp.trigsimp(eq2)
sp.pprint(eq2)

print(" ------------------------------------------------")
print("               dynamic expression 3              ")
print(" ------------------------------------------------")
eq3 = sp.diff( sp.diff(L, dth3), t) - sp.diff(L, th3)
eq3 = sp.trigsimp(eq3)
sp.pprint(eq3)

print(" ------------------------------------------------")
print("               dynamic expression 4              ")
print(" ------------------------------------------------")
eq4 = sp.diff( sp.diff(L, dth4), t) - sp.diff(L, th4)
eq4 = sp.trigsimp(eq4)
sp.pprint(eq4)

ddth1 = sp.diff(th1, t, 2)
ddth2 = sp.diff(th2, t, 2)
ddth3 = sp.diff(th3, t, 2)
ddth4 = sp.diff(th4, t, 2)

unknowns = [ddth1, ddth2, ddth3, ddth4]
eqs = [eq1, eq2, eq3, eq4]

print(" ------------------------------------------------")
print("               dynamic linear form               ")
print(" ------------------------------------------------")

M, rhs = sp.linear_eq_to_matrix(eqs, unknowns) # M * unknowns = rhs
M_simpl = sp.simplify(M)
rhs_simpl = sp.simplify(rhs)

sp.pprint(M_simpl) 
sp.pprint(rhs_simpl)

print(" ------------------------------------------------")
print("          L expression with constraint           ")
print(" ------------------------------------------------")
f_pos = sp.Matrix([
    [-l1*sp.sin(th1) - l2*sp.sin(th2) - l3*sp.sin(th3) - l4*sp.sin(th4)],
    [ l1*sp.cos(th1) + l2*sp.cos(th2) + l3*sp.cos(th3) + l4*sp.cos(th4)]
])
J = f_pos.jacobian([th1, th2, th3, th4])

lda1 = sp.symbols("lambda1")
lda2 = sp.symbols("lambda2")
lda3 = sp.symbols("lambda3")
lda4 = sp.symbols("lambda4")

L = L
sp.pprint(L)
print(" ------------------------------------------------")
print("                    Jaccobian                    ")
print(" ------------------------------------------------")
sp.pprint(J)

print(" ------------------------------------------------")
print("          eq1 expression with constraint         ")
print(" ------------------------------------------------")
eq1 = sp.diff(sp.diff(L, dth1), t) - sp.diff(L, th1) - (J[:,0].T @ sp.Matrix([lda1, lda2]))[0]
eq1 = sp.trigsimp(eq1)
sp.pprint(eq1)
print(" ------------------------------------------------")
print("          eq2 expression with constraint         ")
print(" ------------------------------------------------")
eq2 = sp.diff(sp.diff(L, dth2), t) - sp.diff(L, th2) - (J[:,1].T @ sp.Matrix([lda1, lda2]))[0]
eq2 = sp.trigsimp(eq2)
sp.pprint(eq2)
print(" ------------------------------------------------")
print("          eq3 expression with constraint         ")
print(" ------------------------------------------------")
eq3 = sp.diff(sp.diff(L, dth3), t) - sp.diff(L, th3) - (J[:,2].T @ sp.Matrix([lda1, lda2]))[0]
eq3 = sp.trigsimp(eq3)
sp.pprint(eq3)
print(" ------------------------------------------------")
print("          eq4 expression with constraint         ")
print(" ------------------------------------------------")
eq4 = sp.diff(sp.diff(L, dth4), t) - sp.diff(L, th4) - (J[:,3].T @ sp.Matrix([lda1, lda2]))[0]
eq4 = sp.trigsimp(eq4)
sp.pprint(eq4)

print(" ------------------------------------------------")
print("            ground force constraint              ")
print(" ------------------------------------------------")
ddth1 = sp.diff(th1, t, 2)
ddth2 = sp.diff(th2, t, 2)
ddth3 = sp.diff(th3, t, 2)
ddth4 = sp.diff(th4, t, 2)

eqC = J @ sp.Matrix([ddth1, ddth2, ddth3, ddth4]) + sp.diff(J, t) @ sp.Matrix([dth1, dth2, dth3, dth4])
eqC = sp.trigsimp(eqC)

eq5 = eqC[0]
eq6 = eqC[1]

sp.pprint(eq5) 
sp.pprint(eq6)
print(" ------------------------------------------------")
print("         gnd force solution with constraint      ")
print(" ------------------------------------------------")

unknowns = [ddth1, ddth2, ddth3, ddth4, lda1, lda2]
eqs = [eq1, eq2, eq3, eq4, eq5, eq6]

M, rhs = sp.linear_eq_to_matrix(eqs, unknowns) # M * unknowns = rhs
M_simpl = sp.simplify(M)
rhs_simpl = sp.simplify(rhs)

sp.pprint(M_simpl) 

th1_sym, th2_sym, th3_sym, th4_sym = sp.symbols(
    "th1 th2 th3 th4"
)

dth1_sym, dth2_sym, dth3_sym, dth4_sym = sp.symbols(
    "dth1 dth2 dth3 dth4"
)

ddth1_sym, ddth2_sym, ddth3_sym, ddth4_sym = sp.symbols(
    "ddth1 ddth2 ddth3 ddth4"
)

subs_dict = {
    th1: th1_sym,
    th2: th2_sym,
    th3: th3_sym,
    th4: th4_sym,

    sp.diff(th1,t): dth1_sym,
    sp.diff(th2,t): dth2_sym,
    sp.diff(th3,t): dth3_sym,
    sp.diff(th4,t): dth4_sym,

    sp.diff(th1,t,2): ddth1_sym,
    sp.diff(th2,t,2): ddth2_sym,
    sp.diff(th3,t,2): ddth3_sym,
    sp.diff(th4,t,2): ddth4_sym,
}

rhs_codegen = rhs_simpl.subs(subs_dict)
M_codegen = M_simpl.subs(subs_dict)

printer = NumPyPrinter()

print("--------------------- M -------------------------")
print(printer.doprint(M_codegen))
print("----------------------------------------------")

print("--------------------- rhs -------------------------")
print(printer.doprint(rhs_codegen))
print("----------------------------------------------")


print(" ------------------------------------------------")
print("           dynamic solution with constraint      ")
print(" ------------------------------------------------")
unknowns = [ddth1, ddth2, ddth3, ddth4]
eqs = [eq1, eq2, eq3, eq4]

M, rhs = sp.linear_eq_to_matrix(eqs, unknowns) # M * unknowns = rhs
M_simpl = sp.simplify(M)
rhs_simpl = sp.simplify(rhs)

sp.pprint(M_simpl) 

th1_sym, th2_sym, th3_sym, th4_sym = sp.symbols(
    "th1 th2 th3 th4"
)

dth1_sym, dth2_sym, dth3_sym, dth4_sym = sp.symbols(
    "dth1 dth2 dth3 dth4"
)

ddth1_sym, ddth2_sym, ddth3_sym, ddth4_sym = sp.symbols(
    "ddth1 ddth2 ddth3 ddth4"
)

subs_dict = {
    th1: th1_sym,
    th2: th2_sym,
    th3: th3_sym,
    th4: th4_sym,

    sp.diff(th1,t): dth1_sym,
    sp.diff(th2,t): dth2_sym,
    sp.diff(th3,t): dth3_sym,
    sp.diff(th4,t): dth4_sym,

    sp.diff(th1,t,2): ddth1_sym,
    sp.diff(th2,t,2): ddth2_sym,
    sp.diff(th3,t,2): ddth3_sym,
    sp.diff(th4,t,2): ddth4_sym,
}

rhs_codegen = rhs_simpl.subs(subs_dict)
M_codegen = M_simpl.subs(subs_dict)

printer = NumPyPrinter()

print("--------------------- M -------------------------")
print(printer.doprint(M_codegen))
print("----------------------------------------------")

print("--------------------- rhs -------------------------")
print(printer.doprint(rhs_codegen))
print("----------------------------------------------")

print(" ------------------------------------------------")
print("                  force solution                 ")
print(" ------------------------------------------------")
M = M_simpl
p = -(J @ M.inv() @ J.T).inv() @ J 

sp.pprint(p) 