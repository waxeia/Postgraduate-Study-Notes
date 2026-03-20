import numpy as np
import funs_enght_three as f83

if __name__=="__main__":
    ReD = 5000.0
    N = 20
    E = 1.0
    steps_to_save = (12, 36, 60, 120, 240, 360)

    y, dt, results = f83.couette_cn(
        ReD=ReD, N=N, E=E,
        sum_time_steps=max(steps_to_save),
        output_steps=steps_to_save
    )

    print(f"Δy* = {1 / N:.6f},  Δt* = {dt:.6f}  (E={E}, ReD={ReD})")

    # 为了和课本表格逐行核对：默认只看 num 一列即可
    # 如需解析解对比，把 show_ana=True
    show_ana = True

    for n in steps_to_save:
        t_star, u_num = results[n]
        u_ana = f83.couette_exact(y, t_star, ReD, n_terms=800) if show_ana else None
        print("\n" + "=" * 72)
        print(f"n = {n:4d} steps,  t* = n·Δt* = {t_star:.6f}")
        if show_ana:
            print(" j   y/D      u/u_e (numerical)   u/u_e (exact)")
        else:
            print(" j   y/D      u/u_e")

        for j in range(len(y)):
            y_str = f"{y[j]:.2f}"
            num_s = f83.book_sci(u_num[j], digits=3)

            if show_ana:
                ana_s = f83.book_sci(u_ana[j], digits=3)
                print(f"{j:2d}  {y_str:>4s}     {num_s:>9s}     {ana_s:>9s}")
            else:
                print(f"{j:2d}  {y_str:>4s}     {num_s:>9s}")