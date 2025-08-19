# ------------------------------ LOAD LIBRARIES -------------------------------
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import math

from ngsolve import *

from netgen.meshing import MeshingStep, SetTransformation

from utils import bcolors, coef_fun_grad, vec_grad


# SOLVER
def convergence_torus(maxh, nref, exact, order_u, order_p, vtk_out=None, logs=True):
    """
    Solves the surface Stokes problem with parametric FEM using
    an elliptic reformulation.
    :param maxh: maximal mesh size
    :param nref: refinement level
    :param exact: data structure providing symbolic data for the problem
    :param order_u: polynomial degree for the velocity
    :param order_p: polynomial degree for the pressure
    :param vtk_out: generate vtk output
    :param logs: print solver logs
    :return: total dof, l2-error for u, h1-error for u, l2-error for p, h1-error for p
    """
    order_geo = max(order_u, order_p)

    # mesh torus
    import netgen.occ as occ
    start = time.perf_counter()
    def Curve(t):
        return occ.Pnt(0, 2 + math.cos(t), math.sin(t))

    n = 1000
    pnts = [Curve(2 * math.pi * tdummy / n) for tdummy in range(n + 1)]

    spline = occ.SplineApproximation(points=pnts, tol=1e-12)
    f = occ.Face(occ.Wire(spline))

    torus = f.Revolve(occ.Axis((0, 0, 0), occ.Z), 360)

    mesh = Mesh(occ.OCCGeometry(torus).GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=MeshingStep.MESHSURFACE))
    mesh.Curve(order=order_geo)

    if logs:
        print(f"{bcolors.OKBLUE}Forms defined ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    # import and compile CoefficientFunction's
    start = time.perf_counter()
    coef_u = CoefficientFunction((exact.cfs["u1"], exact.cfs["u2"], exact.cfs["u3"])).Compile()
    coef_p = CoefficientFunction(exact.cfs['p']).Compile()
    coef_f = CoefficientFunction((exact.cfs["f1"], exact.cfs["f2"], exact.cfs["f3"])).Compile()
    coef_g = CoefficientFunction(exact.cfs['g']).Compile()
    coef_nu = CoefficientFunction((exact.cfs["nu1"], exact.cfs["nu2"], exact.cfs["nu3"])).Compile()
    if logs:
        print(f"{bcolors.OKBLUE}CoefficientFunctions compiled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    p_mean = Integrate(coef_p * ds, mesh)

    # FE spaces
    V_nu = VectorH1(mesh, order=order_geo, dirichlet_bbnd="")
    V = VectorH1(mesh, order=order_u, dirichlet_bbnd="")
    Q = H1(mesh, order=order_p, dirichlet_bbnd="")
    N = NumberSpace(mesh)
    fes = V * Q * N

    # grid functions
    gf = GridFunction(fes)
    gfu = gf.components[0]
    gfp = gf.components[1]

    # geometry
    n = specialcf.normal(3)
    Pmat = Id(3) - OuterProduct(n, n)

    gfnu_ho = GridFunction(V_nu)
    with TaskManager():
        gfnu_ho.Interpolate(coef_nu, definedon=mesh.Boundaries(".*"))

    Bmat = Pmat * Grad(gfnu_ho).Trace() * Pmat
    Cmat = Trace(Bmat) * Bmat - Bmat * Bmat

    # bilinear forms:
    (u, p, lam), (v, q, mu) = fes.TnT()
    h = specialcf.mesh_size

    # penalization parameters
    tau = 1.0 / (h * h)

    a = BilinearForm(fes, symmetric=False)

    # velocity diffusion part
    a += InnerProduct(Pmat * Grad(u).Trace() * Pmat - (u * n) * Bmat, Pmat * Grad(v).Trace() * Pmat - (v * n) * Bmat) * ds
    # penalization of the normal component of the velocity
    a += tau * ((u * gfnu_ho) * (v * gfnu_ho)) * ds

    # pressure gradient part
    a += InnerProduct(Pmat * v, grad(p).Trace()) * ds

    # pressure diffusion part
    a += InnerProduct(grad(p).Trace(), grad(q).Trace()) * ds
    a += lam * q * ds

    # new part
    a += (-1.0) * InnerProduct(Cmat * u, grad(q).Trace()) * ds

    # Lagrange multiplier for the pressure mean
    a += p * mu * ds

    # the right-hand side
    f = LinearForm(fes)
    f += InnerProduct(coef_f, Pmat * v) * ds
    f += (coef_f + Pmat * coef_fun_grad(coef_g)) * grad(q).Trace() * ds
    f += p_mean * mu * ds

    if logs:
        print(f"{bcolors.OKBLUE}Forms defined ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    # assembly stage
    start = time.perf_counter()
    with TaskManager():
        prea = Preconditioner(a, "multigrid")
        a.Assemble()
        f.Assemble()

    if logs:
        print(f"{bcolors.OKCYAN}System assembled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    ### LINEAR SOLVER
    start = time.perf_counter()
    with TaskManager():
       solvers.GMRes(A=a.mat, b=f.vec, pre=prea.mat, x=gf.vec, printrates=False, maxsteps=50, reltol=1e-15)

    if logs:
        print(f"{bcolors.OKGREEN}System solved    ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    ### ERRORS

    with TaskManager():
        l2pnorm = sqrt(Integrate(InnerProduct(coef_p, coef_p) * ds, mesh=mesh))
        h1pnorm = sqrt(Integrate(InnerProduct(Pmat * coef_fun_grad(coef_p),
                                              Pmat * coef_fun_grad(coef_p)) * ds, mesh=mesh))
        l2unorm = sqrt(Integrate(InnerProduct(Pmat * coef_u, Pmat * coef_u) * ds, mesh=mesh))
        h1unorm = sqrt(Integrate(InnerProduct(Pmat * vec_grad(coef_u) * Pmat,
                                              Pmat * vec_grad(coef_u) * Pmat) * ds, mesh=mesh))
        l2p = sqrt(Integrate(InnerProduct(gfp - coef_p, gfp - coef_p) * ds, mesh=mesh)) / l2pnorm
        h1p = sqrt(Integrate(InnerProduct(grad(gfp).Trace() - Pmat * coef_fun_grad(coef_p),
                                          grad(gfp).Trace() - Pmat * coef_fun_grad(coef_p)) * ds, mesh=mesh)) / h1pnorm
        l2u = sqrt(Integrate(InnerProduct(Pmat * (gfu - coef_u), Pmat * (gfu - coef_u)) * ds, mesh=mesh)) / l2unorm
        h1u = sqrt(Integrate(InnerProduct(Pmat * (Grad(gfu).Trace() - vec_grad(coef_u)) * Pmat - ((gfu - coef_u) * n) * Bmat,
                                          Pmat * (Grad(gfu).Trace() - vec_grad(coef_u)) * Pmat - ((gfu - coef_u) * n) * Bmat) * ds, mesh=mesh)) / h1unorm

    # optional VTK output
    if vtk_out:
        with TaskManager():
            vtk = VTKOutput(ma=mesh,
                            coefs=[gfu, gfp, coef_u, coef_p],
                            names=["u", "p", "coef_u", "coef_p"],
                            filename=f"output/vtk/torus_p{order_u}-p{order_p}_{nref}",
                            subdivision=order_geo)
            vtk.Do(vb=BND)

    return V.ndof + Q.ndof, l2u, h1u, l2p, h1p


def figure_tooth(maxh, nref, exact, order_u, order_p, logs=False):
    order_geo = max(order_u, order_p)

    import netgen.csg as csg
    start = time.perf_counter()
    geo = csg.CSGeometry()
    geo.Add(csg.Sphere(csg.Pnt(0, 0, 0), 1))
    mesh = Mesh(geo.GenerateMesh(maxh=maxh, optsteps2d=3, perfstepsend=MeshingStep.MESHSURFACE))
    mesh.Curve(order=order_geo)

    theta, phi = atan2(y, x), acos(z)
    cf_def = CoefficientFunction((sin(phi) * cos(theta) + cos(phi)**2, sin(phi) * sin(theta), cos(phi)))
    cf_id = CoefficientFunction((x, y, z))

    Vmesh = VectorH1(mesh, order=order_geo, dirichlet_bbnd="")

    gf_def = GridFunction(Vmesh)
    with TaskManager():
        gf_def.Interpolate(cf_def - cf_id, definedon=mesh.Boundaries(".*"))
        mesh.SetDeformation(gf_def)
    print(f"{bcolors.OKBLUE}Mesh generated ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    start = time.perf_counter()
    coef_f = CoefficientFunction((x*y, sin(y**2 * x), -exp(-z))).Compile()
    coef_g = CoefficientFunction(0).Compile()
    coef_nu = CoefficientFunction((exact.cfs["nu1"], exact.cfs["nu2"], exact.cfs["nu3"])).Compile()
    print(f"{bcolors.OKBLUE}CoefficientFunctions compiled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    p_mean = 0.0

    V_nu = VectorH1(mesh, order=order_geo, dirichlet_bbnd="")
    V = VectorH1(mesh, order=order_u, dirichlet_bbnd="")
    Q = H1(mesh, order=order_p, dirichlet_bbnd="")
    N = NumberSpace(mesh)
    fes = V * Q * N

    # grid functions
    gf = GridFunction(fes)
    gfu = gf.components[0]
    gfp = gf.components[1]

    # geometry
    n = specialcf.normal(3)
    Pmat = Id(3) - OuterProduct(n, n)

    gfnu_ho = GridFunction(V_nu)
    with TaskManager():
        gfnu_ho.Set(coef_nu, definedon=mesh.Boundaries(".*"))

    Bmat = Pmat * Grad(gfnu_ho) * Pmat
    Cmat = Trace(Bmat) * Bmat - Bmat * Bmat

    # bilinear forms:
    (u, p, lam), (v, q, mu) = fes.TnT()
    h = specialcf.mesh_size

    # penalization parameters
    tau = 1.0 / (h * h)
    alpha = h * h

    a = BilinearForm(fes, symmetric=False)

    # velocity diffusion part
    a += alpha * InnerProduct(Pmat * u, Pmat * v) * ds
    a += InnerProduct(Pmat * Sym(Grad(u).Trace()) * Pmat - (u * n) * Bmat,
                      Pmat * Sym(Grad(v).Trace()) * Pmat - (v * n) * Bmat) * ds
    # penalization of the normal component of the velocity
    a += tau * ((u * gfnu_ho) * (v * gfnu_ho)) * ds

    # pressure gradient part
    a += InnerProduct(Pmat * v, grad(p).Trace()) * ds

    # pressure diffusion part
    a += InnerProduct(grad(p).Trace(), grad(q).Trace()) * ds
    a += lam * q * ds

    # new part
    a += (-2.0) * InnerProduct(Cmat * u, grad(q).Trace()) * ds

    a += p * mu * ds

    f = LinearForm(fes)
    f += InnerProduct(Pmat * coef_f, Pmat * v) * ds
    f += (Pmat * coef_f + 2 * Pmat * coef_fun_grad(coef_g)) * grad(q).Trace() * ds
    f += alpha * coef_g * q * ds
    f += p_mean * mu * ds

    print(f"{bcolors.OKBLUE}Forms defined ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    start = time.perf_counter()
    with TaskManager():
        prea = Preconditioner(a, "multigrid")
        a.Assemble()
        f.Assemble()

    if logs:
        print(f"{bcolors.OKCYAN}System assembled ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    ### LINEAR SOLVER

    start = time.perf_counter()
    with TaskManager():
        solvers.GMRes(A=a.mat, b=f.vec, pre=prea.mat, x=gf.vec, printrates=False, maxsteps=50, reltol=1e-15)

    if logs:
        print(f"{bcolors.OKBLUE}System solved    ({time.perf_counter() - start:.5f} s).{bcolors.ENDC}")

    print(f"{V.ndof + Q.ndof} dofs.")

    ### VTK Output

    with TaskManager():
        vtk = VTKOutput(ma=mesh,
                        coefs=[gfu, gfp],
                        names=["u", "p"],
                        filename=f"output/vtk/tooth_p{order_u}-p{order_p}_{nref}",
                        subdivision=order_geo)
        vtk.Do(vb=BND)

    return