import sympy as sp  
from sympy.printing.numpy import NumPyPrinter

t = sp.symbols("t")
m0 = sp.symbols("m0")
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

x0 = sp.Function("x0")(t)
y0 = sp.Function("y0")(t)

x1 = x0 - l1 * sp.sin(th1)
y1 = y0 - l1 * sp.cos(th1)

x2 = x1 - l2 * sp.sin(th2)
y2 = y1 - l2 * sp.cos(th2)

x3 = x0 - l3 * sp.sin(th3)
y3 = y0 - l3 * sp.cos(th3)

x4 = x3 - l4 * sp.sin(th4)
y4 = y3 - l4 * sp.cos(th4)

dx0 = sp.diff(x0, t)
dy0 = sp.diff(y0, t)

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

print("---------------------------------------------------------------")
print("                dth1 dth2 dth3 dth4 expression                 ")
print("---------------------------------------------------------------")
sp.pprint(dth1)
sp.pprint(dth2)
sp.pprint(dth3)
sp.pprint(dth4)

print("---------------------------------------------------------------")
print("                         dx0 expression                        ")
print("---------------------------------------------------------------")
sp.pprint(dx0)

print("---------------------------------------------------------------")
print("                         dx1 expression                        ")
print("---------------------------------------------------------------")
sp.pprint(dx1)

print("---------------------------------------------------------------")
print("                         dx2 expression                        ")
print("---------------------------------------------------------------")
sp.pprint(dx2)

print("---------------------------------------------------------------")
print("                         dx3 expression                        ")
print("---------------------------------------------------------------")
sp.pprint(dx3)

print("---------------------------------------------------------------")
print("                         dx4 expression                        ")
print("---------------------------------------------------------------")
sp.pprint(dx4)

print("---------------------------------------------------------------")
print("                         dy0 expression                        ")
print("---------------------------------------------------------------")
sp.pprint(dy0)

print("---------------------------------------------------------------")
print("                         dy1 expression                        ")
print("---------------------------------------------------------------")
sp.pprint(dy1)

print("---------------------------------------------------------------")
print("                         dy2 expression                        ")
print("---------------------------------------------------------------")
sp.pprint(dy2)

print("---------------------------------------------------------------")
print("                         dy3 expression                        ")
print("---------------------------------------------------------------")
sp.pprint(dy3)

print("---------------------------------------------------------------")
print("                         dy4 expression                        ")
print("---------------------------------------------------------------")
sp.pprint(dy4)

print("---------------------------------------------------------------")
print("                          T expression                         ")
print("---------------------------------------------------------------")
T = 1/2*m0*(dx0**2 + dy0**2) + 1/2*m1*(dx1**2 + dy1**2) + 1/2*m2*(dx2**2 + dy2**2) + 1/2*m3*(dx3**2 + dy3**2) + 1/2*m4*(dx4**2 + dy4**2)
sp.pprint(T)

print("---------------------------------------------------------------")
print("                          V expression                         ")
print("---------------------------------------------------------------")
V = m0*g*y0 + m1*g*y1 + m2*g*y2 + m3*g*y3 + m4*g*y4
sp.pprint(V)

print("---------------------------------------------------------------")
print("                          L expression                         ")
print("---------------------------------------------------------------")
L = T - V
sp.pprint(L)

print("---------------------------------------------------------------")
print("                    dymanic expression 0-x                     ")
print("---------------------------------------------------------------")

tau1 = sp.symbols("τ1")
tau2 = sp.symbols("τ2")
tau3 = sp.symbols("τ3")
tau4 = sp.symbols("τ4")

eq0x = sp.diff( sp.diff(L, dx0), t ) - sp.diff(L, x0)
eq0x = sp.trigsimp(eq0x)
sp.pprint(eq0x)

print("---------------------------------------------------------------")
print("                    dynamic expression 0-y                     ")
print("---------------------------------------------------------------")
eq0y = sp.diff( sp.diff(L, dy0), t ) - sp.diff(L, y0)
eq0y = sp.trigsimp(eq0y)
sp.pprint(eq0y)

print("---------------------------------------------------------------")
print("                    dynamic expression th1                     ")
print("---------------------------------------------------------------")
eq1 = sp.diff( sp.diff(L, dth1), t) - sp.diff(L, th1) - tau1
eq1 = sp.trigsimp(eq1)
sp.pprint(eq1)

print("---------------------------------------------------------------")
print("                    dynamic expression th2                     ")
print("---------------------------------------------------------------")
eq2 = sp.diff( sp.diff(L, dth2), t) - sp.diff(L, th2) - tau2
eq2 = sp.trigsimp(eq2)
sp.pprint(eq2)

print("---------------------------------------------------------------")
print("                   dynamic expression th3                      ")
print("---------------------------------------------------------------")
eq3 = sp.diff( sp.diff(L, dth3), t) - sp.diff(L, th3) - tau3
eq3 = sp.trigsimp(eq3)
sp.pprint(eq3)

print("---------------------------------------------------------------")
print("                   dynamic expression th4                      ")
print("---------------------------------------------------------------")
eq4 = sp.diff( sp.diff(L, dth4), t) - sp.diff(L, th4) - tau4
eq4 = sp.trigsimp(eq4)
sp.pprint(eq4)

print("---------------------------------------------------------------")
print("                     dynamic linear form                       ")
print("---------------------------------------------------------------")
ddx0 = sp.diff(x0, t, 2)
ddy0 = sp.diff(y0, t, 2)
ddth1 = sp.diff(th1, t, 2)
ddth2 = sp.diff(th2, t, 2)
ddth3 = sp.diff(th3, t, 2)
ddth4 = sp.diff(th4, t, 2)

unknowns = [ddx0, ddy0, ddth1, ddth2, ddth3, ddth4]
eqs = [eq0x, eq0y, eq1, eq2, eq3, eq4]

M, rhs = sp.linear_eq_to_matrix(eqs, unknowns) # M * unknowns = rhs
M_simpl = sp.simplify(M)
rhs_simpl = sp.simplify(rhs)

sp.pprint(M_simpl)
sp.pprint(rhs_simpl)

print("---------------------------------------------------------------")
print("                       dynamic w/o constraint                  ")
print("---------------------------------------------------------------")

unknowns = [ddx0, ddy0, ddth1, ddth2, ddth3, ddth4]
eqs = [eq0x, eq0y, eq1, eq2, eq3, eq4]

x0_sym, y0_sym, th1_sym, th2_sym, th3_sym, th4_sym = sp.symbols(
    "x0, y0, th1 th2 th3 th4"
)

dx0_sym, dy0_sym, dth1_sym, dth2_sym, dth3_sym, dth4_sym = sp.symbols(
    "dx0, dy0, dth1 dth2 dth3 dth4"
)

ddx0_sym, ddy0_sym, ddth1_sym, ddth2_sym, ddth3_sym, ddth4_sym = sp.symbols(
    "ddx0, ddy0, ddth1 ddth2 ddth3 ddth4"
)

subs_dict = {
    x0 : x0_sym,
    y0 : y0_sym,
    th1: th1_sym,
    th2: th2_sym,
    th3: th3_sym,
    th4: th4_sym,

    sp.diff(x0,t)  : dx0_sym,
    sp.diff(y0,t)  : dy0_sym,
    sp.diff(th1,t) : dth1_sym,
    sp.diff(th2,t) : dth2_sym,
    sp.diff(th3,t) : dth3_sym,
    sp.diff(th4,t) : dth4_sym,

    sp.diff(x0, t, 2) : ddx0_sym,
    sp.diff(y0, t, 2) : ddy0_sym,
    sp.diff(th1, t, 2) : ddth1_sym,
    sp.diff(th2, t, 2) : ddth2_sym,
    sp.diff(th3, t, 2) : ddth3_sym,
    sp.diff(th4, t, 2) : ddth4_sym
}

rhs_codegen = rhs_simpl.subs(subs_dict)
M_codegen = M_simpl.subs(subs_dict)

printer = NumPyPrinter()

print("---------------------------------------------------------------")
print("                   M code gen w/o constraint                   ")
print("---------------------------------------------------------------")

print(printer.doprint(M_codegen))

print("---------------------------------------------------------------")
print("                  rhs code gen w/o constraint                  ")
print("---------------------------------------------------------------")

print(printer.doprint(rhs_codegen))

print("---------------------------------------------------------------")
print("                J1 f_pos expression with constraint            ")
print("---------------------------------------------------------------")
x0_pos = sp.Matrix([
    [x0],
    [y0]
])
f1_pos = sp.Matrix([
    [x0 - l1*sp.sin(th1) - l2*sp.sin(th2)],
    [y0 - l1*sp.cos(th1) - l2*sp.cos(th2)]
])
f2_pos = sp.Matrix([
    [x0 - l3*sp.sin(th3) - l4*sp.sin(th4)],
    [y0 - l3*sp.cos(th3) - l4*sp.cos(th4)]
])

J_x0pos = x0_pos.jacobian([x0, y0, th1, th2, th3, th4])
J_f1pos = f1_pos.jacobian([x0, y0, th1, th2, th3, th4])
J_f2pos = f2_pos.jacobian([x0, y0, th1, th2, th3, th4])

sp.pprint(J_f1pos)

print("---------------------------------------------------------------")
print("                J2 f_pos expression with constraint            ")
print("---------------------------------------------------------------")
sp.pprint(J_f2pos)

print("---------------------------------------------------------------")
print("              d_J2 f_pos expression with constraint            ")
print("---------------------------------------------------------------")
dJ2 = sp.diff(J_f2pos, t)
sp.pprint(dJ2)

print("---------------------------------------------------------------")
print("                    d_J expression with constraint             ")
print("---------------------------------------------------------------")
Jc = sp.Matrix.vstack(J_x0pos, J_f1pos, J_f2pos)
dJc = sp.diff(Jc, t)
sp.pprint(dJc)

print("---------------------------------------------------------------")
print("                J0 x0_pos expression with constraint           ")
print("---------------------------------------------------------------")
sp.pprint(J_x0pos)

print("---------------------------------------------------------------")
print("                   eq1 expression with constraint              ")
print("---------------------------------------------------------------")

lda1 = sp.symbols("lambda1")
lda2 = sp.symbols("lambda2")
lda3 = sp.symbols("lambda3")
lda4 = sp.symbols("lambda4")

tau1 = sp.symbols("τ1")
tau2 = sp.symbols("τ2")
tau3 = sp.symbols("τ3")
tau4 = sp.symbols("τ4")

eq1 = sp.diff(sp.diff(L, dth1), t) - sp.diff(L, th1) - (J_f1pos[:,2].T @ sp.Matrix([lda1, lda2]))[0] - tau1 
eq1 = sp.trigsimp(eq1)
sp.pprint(eq1)

print("---------------------------------------------------------------")
print("                   eq2 expression with constraint              ")
print("---------------------------------------------------------------")

eq2 = sp.diff(sp.diff(L, dth2), t) - sp.diff(L, th2) - (J_f1pos[:,3].T @ sp.Matrix([lda1, lda2]))[0] - tau2
eq2 = sp.trigsimp(eq2)
sp.pprint(eq2)

print("---------------------------------------------------------------")
print("                   eq3 expression with constraint              ")
print("---------------------------------------------------------------")

eq3 = sp.diff(sp.diff(L, dth3), t) - sp.diff(L, th3) - (J_f2pos[:,4].T @ sp.Matrix([lda3, lda4]))[0] - tau3
eq3 = sp.trigsimp(eq3)
sp.pprint(eq3)

print("---------------------------------------------------------------")
print("                   eq4 expression with constraint              ")
print("---------------------------------------------------------------")

eq4 = sp.diff(sp.diff(L, dth4), t) - sp.diff(L, th4) - (J_f2pos[:,5].T @ sp.Matrix([lda3, lda4]))[0] - tau4
eq4 = sp.trigsimp(eq4)
sp.pprint(eq4)

print("---------------------------------------------------------------")
print("                   ground force constraint                     ")
print("---------------------------------------------------------------")

eqC1 = J_f1pos @ sp.Matrix([ddx0, ddy0, ddth1, ddth2, ddth3, ddth4]) + sp.diff(J_f1pos, t) @ sp.Matrix([dx0, dy0, dth1, dth2, dth3, dth4])
eqC1 = sp.trigsimp(eqC1)

eqC2 = J_f2pos @ sp.Matrix([ddx0, ddy0, ddth1, ddth2, ddth3, ddth4]) + sp.diff(J_f2pos, t) @ sp.Matrix([dx0, dy0, dth1, dth2, dth3, dth4])
eqC2 = sp.trigsimp(eqC2)

eq5 = eqC1[0]
eq6 = eqC1[1]
eq7 = eqC2[0]
eq8 = eqC2[1]

sp.pprint(eq5)
sp.pprint(eq6)
sp.pprint(eq7)
sp.pprint(eq8)

print("---------------------------------------------------------------")
print("              ground force solution with constraint            ")
print("---------------------------------------------------------------")

unknowns = [ddx0, ddy0, ddth1, ddth2, ddth3, ddth4, lda1, lda2, lda3, lda4]
eqs = [eq0x, eq0y, eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]

M, rhs = sp.linear_eq_to_matrix(eqs, unknowns) # M * unknowns = rhs
M_simpl = sp.simplify(M)
rhs_simpl = sp.simplify(rhs)

sp.pprint(M_simpl)
sp.pprint(rhs_simpl)

x0_sym, y0_sym, th1_sym, th2_sym, th3_sym, th4_sym = sp.symbols(
    "x0, y0, th1 th2 th3 th4"
)

dx0_sym, dy0_sym, dth1_sym, dth2_sym, dth3_sym, dth4_sym = sp.symbols(
    "dx0, dy0, dth1 dth2 dth3 dth4"
)

ddx0_sym, ddy0_sym, ddth1_sym, ddth2_sym, ddth3_sym, ddth4_sym = sp.symbols(
    "ddx0, ddy0, ddth1 ddth2 ddth3 ddth4"
)

subs_dict = {
    x0 : x0_sym,
    y0 : y0_sym,
    th1: th1_sym,
    th2: th2_sym,
    th3: th3_sym,
    th4: th4_sym,

    sp.diff(x0,t)  : dx0_sym,
    sp.diff(y0,t)  : dy0_sym,
    sp.diff(th1,t) : dth1_sym,
    sp.diff(th2,t) : dth2_sym,
    sp.diff(th3,t) : dth3_sym,
    sp.diff(th4,t) : dth4_sym,

    sp.diff(x0, t, 2) : ddx0_sym,
    sp.diff(y0, t, 2) : ddy0_sym,
    sp.diff(th1, t, 2) : ddth1_sym,
    sp.diff(th2, t, 2) : ddth2_sym,
    sp.diff(th3, t, 2) : ddth3_sym,
    sp.diff(th4, t, 2) : ddth4_sym
}

rhs_codegen = rhs_simpl.subs(subs_dict)
M_codegen = M_simpl.subs(subs_dict)

printer = NumPyPrinter()

print("---------------------------------------------------------------")
print("                          M code gen                           ")
print("---------------------------------------------------------------")

print(printer.doprint(M_codegen[:6,:6]))

print("---------------------------------------------------------------")
print("                        rhs code gen                           ")
print("---------------------------------------------------------------")

print(printer.doprint(rhs_codegen[:6,:6]))