#import "@preview/cetz:0.5.2"
#import "@preview/cetz-plot:0.1.4": plot
#import "@preview/fontawesome:0.5.0": *
#import "@preview/fletcher:0.5.8" as fletcher: edge, node
#import "@preview/touying:0.6.1": *
#import "@local/touying-simpl-uni:1.0.0": *

// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#show: ecnu-theme.with(
  // Lang and font configuration
  lang: "en",
  font: "Libertinus Serif",

  // Basic information
  config-info(
    title: [How to solve the double pendulum efficently],
    short-title: [Adaptive Runge-Kutta],
    subtitle: [Adaptive Runge-Kutta],
    author: [Jorge Munoz Laredo],
    date: datetime.today(),
    institution: [National University of Engineering],
  ),
)
#title-slide()

== The double pendulum key facts

#grid(
  columns: (2fr, 1fr),
  // Left is 1 unit wide, right is 2 units wide
  gutter: 1em,
  // Space between columns
  [
    With the position of each mass you can calculate the lagragian
    $
      x_1 = L_1 sin theta_1 and y_1 = - L_1 cos theta_1 \
      x_2 = x_1 + L_2 sin theta_2 and y_2 = y_1 - L_2 cos theta_2
    $
    The phase space is four dimensional $(theta_1, theta_2, dot(theta)_1, dot(theta)_2)$
    $ cal(L) = T - U $
    Using the _Euler Lagrange_ equations you obtain:
    $
      dot.double(theta)_1 = f_1(theta_1, theta_2, dot(theta)_1, dot(theta)_2) \
      dot.double(theta)_2 = f_2(theta_1, theta_2, dot(theta)_1, dot(theta)_2)
    $
  ],
  [
    #figure(
      image("images/dp.webp", width: 120%),
      caption: [Double Pendulum formulation],
    )
  ],
)

== Motivating Kutta

Differential equation: $y' = f(t, y)= t^2 + y^2$, $y(0) = 0.46$ (Riccati Equation)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  [
    #figure(
      image("images/explicit.png"),
      caption: [Improved Euler $y_(n+1) = y_n + h/2(k_1 + k_2)$],
    )
  ],
  [
    #figure(
      image("images/mid.png"),
      caption: [Midpoint Runge Kutta
        $y_n+h k_2$],
    )
  ],
  [
    #figure(
      image("images/rk.png"),
      caption: [Runge Kutta Order $3$],
    )
  ],
)

== From Taylor series to Runge Kutta

#set par(spacing: 0.55em)

#grid(
  columns: (1.4fr, 1fr),
  gutter: 1em,
  [
    Using this motivation Runge and Kutta propose the follow form to obtain the solution.
    $ k_i = f(t_n + c_i h, y_n + h sum_j a_(i j) k_j) $

    $ y_(n+1) = y_n + h sum_i b_i k_i $

    _How do you obtain coefficients $(a_(i j) , b_i, c_i)$?_

    $ sum_(j=1) a_(i j) = c_i $
    Using the *Taylor Polinomial* // expansion of $y(t+h)$, $y'=f(t, y)$ you asure:
    $ y_1 - y(t_0 + h) = cal(O)(h^(p+1)) text("as") h -> 0 $

  ],
  [
    #let bf(x) = math.bold(math.upright(x))

    We have second derivatives; use vectors instead of scalars.

    $ bf(u)=(theta_1, dot(theta)_1, theta_2, dot(theta)_1) $

    $ dot(bf(u))=(dot(theta)_1, dot.double(theta)_1, dot(theta)_2, dot.double(theta)_2) $

    Therefore:

    $ bf(u_(n+1)) = bf(u_n) + h sum_i b_i bf(k_i) $

    #figure(
      image("images/table.png", width: 60%),
      caption: [RK coefficients],
    )
  ],
)

== Working example with  RK4 and RK8

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #figure(
      image("images/rk4_t.png", width: 80%),
      caption: [Butcher tableau for RK4],
    )
  ],
  [
    #figure(
      table(
        columns: (auto, auto, auto),
        [Order], [Stages], [Total coefficients],
        [8], [13], [80],
      ),
      caption: [Dormand & Prince (1981)],
    )
    - Fixed-step variant: adaptive control disabled, $h$ held constant.
    - Coefficients loaded directly from `scipy.integrate.DOP853`.
  ],
)

#v(1cm)

#tblock(title: [Global truncation error])[
  Scales as $cal(O)(h^4)$ for RK4 and $cal(O)(h^8)$ for RK8 — for the same $h$, RK8 error is roughly $h^4$ times smaller, which for $h=0.01$ is a factor of $10^8$.
]

== Energy Drift Comparison

#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 1.2em,
  [
    The Hamiltonian $H$ is conserved exactly — any drift is *pure numerical error* accumulating over time.
    $ Delta H(t) = H(t) - H(0) $

    #v(0.7em)

    #figure(
      cetz-canvas({
        import cetz.draw: *

        let W = 7.5
        let H = 5.2

        // Grid
        for i in range(1, 5) {
          line((i * W / 4, 0), (i * W / 4, H), stroke: (paint: luma(220), thickness: 0.3pt))
        }
        for i in range(1, 3) {
          line((0, i * H / 3), (W, i * H / 3), stroke: (paint: luma(220), thickness: 0.3pt))
        }

        // Axes
        line((0, 0), (W, 0), mark: (end: ">"))
        line((0, 0), (0, H), mark: (end: ">"))
        content((W + 0.3, 0), text(fill: black, size: 10pt)[$t$])
        content((0.05, H + 0.25), text(fill: black, size: 10pt)[$Delta H$])
        content((0, -0.25), $0$)

        // Euler — exponential blow-up
        bezier(
          (0, 0.05),
          (W * 0.55, H * 0.95),
          (W * 0.15, 0.15),
          (W * 0.4, H * 0.7),
          stroke: (paint: red, thickness: 1.5pt),
        )
        content((W * 0.54, H * 0.95 + 0.22), text(fill: red, size: 8pt)[Euler])

        // RK4 — gentle polynomial drift
        bezier(
          (0, 0.02),
          (W * 0.92, H * 0.48),
          (W * 0.35, 0.08),
          (W * 0.7, H * 0.3),
          stroke: (paint: blue, thickness: 1.5pt),
        )
        content((W * 0.94, H * 0.48 + 0.22), text(fill: blue, size: 12pt)[RK4])

        // RK8 — near flat
        bezier(
          (0, 0.01),
          (W * 0.92, H * 0.06),
          (W * 0.4, 0.02),
          (W * 0.7, 0.055),
          stroke: (paint: green, thickness: 1.5pt),
        )
        content((W * 0.94, H * 0.06 + 0.22), text(fill: green, size: 12pt)[RK8])
      }),
      caption: [Energy drift over time for different integrators],
    )
  ],
  [
    #figure(
      table(
        columns: (auto, auto, auto),
        stroke: 0.5pt,
        table.header([*Method*], [*Drift behaviour*], [*Order*]),
        [Euler], [exponential growth], [1],
        [RK4], [polynomial drift], [4],
        [RK8], [near machine $epsilon$], [8],
      ),
      caption: [Energy drift vs. method order],
    )

    #v(2.7em)

    _Energy drift *confirms method order* without needing a reference solution._
  ],
)

== Working example with RK4 and RK8

#figure(
  image("images/video.png", width: 70%),
  caption: [Comparison of RK4 and RK8 solutions with $h=0.001$],
)

== Plots

#grid(
  columns: (1fr, 1fr),
  [
    #figure(
      image("images/rk8_0.01_0.005_0.002_0.001_convergence_ld.pdf", width: 100%),
      caption: [],
    )

  ],
  [
    #figure(
      image("images/rk8_0.01_0.0075_0.005_0.0035_0.002_0.001_convergence.pdf", width: 100%),
      caption: [],
    )
  ],
)

== Adaptive step-size control

#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 1.2em,
  [
    Fixed $h$ wastes work in smooth regions and loses accuracy in dynamic ones.
    *Goal:* keep the local error near a tolerance $epsilon$ by adjusting $h$ at every step.

    At step $n$, compute two estimates of $y(t_(n+1))$:
    $
             u_(n+1) & : quad "order" p quad   & ("solution kept") \
      tilde(u)_(n+1) & : quad "order" p+1 quad & ("error checker")
    $
    Local error estimate and its power-law scaling:
    $
      E_n (h) = lr(||u_(n+1) - tilde(u)_(n+1)||) approx C h^(p+1)
    $
    Require $E_n (h_"new") = epsilon$, solve for scale factor $q = h_"new" / h$:
    $
      q^(p+1) dot E_n (h) approx epsilon
      quad => quad
      q = lr((frac(epsilon, E_n (h))))^(1\/(p+1))
    $
    $h_"new" = q dot h$ — *accepted* if $q >= 1$, *rejected & retried* if $q < 1$.
  ],
  [
    #align(center)[
      #figure(
        cetz-canvas({
          import cetz.draw: *
          let W = 9.0
          let Hc = 4.2

          // Axes
          line((0, -0.6), (W, -0.6), mark: (end: ">"), stroke: 1.2pt)
          line((0, -0.6), (0, Hc), mark: (end: ">"), stroke: 1.2pt)
          content((W + 0.35, -0.6), text(size: 11pt)[$t$])
          content((-0.35, Hc + 0.08), text(size: 11pt)[$y$])

          // Curve: smooth — dynamic burst — smooth
          let x1 = W * 0.4
          let x2 = W * 0.64
          bezier(
            (0, 0.9),
            (x1, 1.05),
            (W * 0.1, 0.7),
            (W * 0.3, 1.3),
            stroke: (paint: black, thickness: 1.8pt),
          )
          bezier(
            (x1, 1.05),
            (x2, 0.95),
            (W * 0.47, 2.3),
            (W * 0.58, -0.15),
            stroke: (paint: black, thickness: 1.8pt),
          )
          bezier(
            (x2, 0.95),
            (W, 1.1),
            (W * 0.74, 0.65),
            (W * 0.88, 1.45),
            stroke: (paint: black, thickness: 1.8pt),
          )

          // Step markers: wide — narrow — wide
          let marks = (0.0, 0.85, 1.7, 2.1, 2.32, 2.5, 2.65, 2.78, 2.9, 3.4, 4.15, 5.0)
          for x in marks {
            let sx = x * W / 5.0
            line((sx, -0.75), (sx, -0.45), stroke: (thickness: 1.3pt))
          }

          // Region labels below axis
          content((W * 0.17, -1.1), text(size: 14pt)[large $h$])
          content((W * 0.5, -1.1), text(size: 14pt)[small $h$])
          content((W * 0.82, -1.1), text(size: 14pt)[large $h$])

          // Region labels on the curve
          content((W * 0.17, Hc * 0.75), text(size: 14pt, fill: gray)[smooth])
          content((W * 0.5, Hc * 0.85), text(size: 14pt, fill: red)[dynamic])
          content((W * 0.82, Hc * 0.75), text(size: 14pt, fill: gray)[smooth])
        }),
        caption: [Adaptive step sizing: wide steps in smooth regions, narrow in dynamic ones],
      )
    ]
  ],
)

#tblock(title: [Embedded pair — no extra cost])[
  Both $u_(n+1)$ and $tilde(u)_(n+1)$ are built from the *same stages $k_i$* — the error check costs zero extra RHS evaluations.
]

== Embedding Runge Kutta

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    A single Butcher tableau stores *two methods at once* by carrying two rows of weights.
    The stages $k_i$ are computed once; both solutions are cheap linear combinations:
    $
             u_(n+1) & = u_n + h sum_i b_i k_i quad      &   arrow "order" p \
      tilde(u)_(n+1) & = u_n + h sum_i hat(b)_i k_i quad & arrow "order" p+1
    $
    Error estimate — no extra $f$ calls:
    $ E_n = h lr(|| sum_i (b_i - hat(b)_i) k_i ||) $

    *FSAL — First Same As Last:* the final stage $k_s$ satisfies
    $ k_s = f(t_(n+1), u_(n+1)) = k_1 "of step" n+1 $
    so it is *reused*, saving one RHS evaluation per accepted step.
  ],
  [
    // Butcher tableau schematic
    #align(center)[
      #figure(
        cetz-canvas({
          import cetz.draw: *
          let aw = 4.5 // A block width
          let ah = 2.8 // A block height
          let bh = 0.65 // b-row height
          let cw = 0.8 // c column width

          // c block
          rect((-cw, 0), (0, -ah), fill: luma(245), stroke: (paint: luma(160), thickness: 0.7pt))
          content((-cw * 0.5, -ah * 0.5), text(size: 11pt)[$bold(c)$])

          // A block (lower triangular, gray)
          rect((0, 0), (aw, -ah), fill: luma(232), stroke: (paint: luma(160), thickness: 0.7pt))
          // Triangle outline inside A to hint at lower-triangular structure
          line((0, 0), (0, -ah), stroke: (paint: luma(150), thickness: 0.5pt))
          line((0, -ah), (aw, -ah), stroke: (paint: luma(150), thickness: 0.5pt))
          line((0, 0), (aw, -ah), stroke: (paint: luma(150), dash: "dashed", thickness: 0.8pt))
          content((aw * 0.35, -ah * 0.5), text(size: 11pt)[$A$])
          content((aw * 0.78, -ah * 0.18), text(size: 9pt, fill: luma(140))[0])

          // Thick separator between A and b rows
          line((-cw, -ah), (aw, -ah), stroke: (paint: black, thickness: 1.4pt))
          // Vertical separator between c/0 and A/b
          line((0, 0.15), (0, -ah - 2 * bh), stroke: (paint: luma(120), thickness: 0.7pt))

          // b row — order p (blue)
          rect((-cw, -ah), (0, -ah - bh), fill: luma(245), stroke: (paint: luma(160), thickness: 0.6pt))
          rect((0, -ah), (aw, -ah - bh), fill: rgb("#ddeeff"), stroke: (paint: blue, thickness: 0.7pt))
          content((aw * 0.5, -ah - bh * 0.5), text(size: 11pt, fill: blue)[$bold(b)^top quad$ order $p$])

          // b-hat row — order p+1 (red)
          rect((-cw, -ah - bh), (0, -ah - 2 * bh), fill: luma(245), stroke: (paint: luma(160), thickness: 0.6pt))
          rect((0, -ah - bh), (aw, -ah - 2 * bh), fill: rgb("#ffdddd"), stroke: (paint: red, thickness: 0.7pt))
          content((aw * 0.5, -ah - bh * 1.5), text(size: 11pt, fill: red)[$hat(bold(b))^top$ order $p+1$])
        }),
        caption: [Butcher tableau for an embedded Runge–Kutta pair],
      )
    ]

    #v(0.6em)
    #figure(
      table(
        columns: (auto, auto, auto, auto, auto),
        stroke: 0.5pt,
        table.header([*Pair*], [*Orders*], [*Stages*], [*FSAL*], [*Evals\/step*]),
        [RK45], [$5(4)$], [6], [yes], [5],
        [DOP853], [$8(5)$], [12], [yes], [11],
      ),
      caption: [FSAL saves one evaluation on every accepted step],
    )
  ],
)

#tblock(title: [Why DOP853 over RK45 for the double pendulum?])[
  At equal wall-clock cost, DOP853 takes larger steps and commits far less error per unit of compute — critical for a chaotic system where local errors grow exponentially.
]

== Testing out Adaptive RK45 vs DOP853

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    #figure(
      table(
        columns: (auto, auto, auto),
        stroke: 0.5pt,
        table.header([], [*RK45*], [*DOP853*]),
        [*Accepted steps*], [1 579], [633],
        [$h_min$], [$1.0 times 10^(-3)$], [$1.7 times 10^(-2)$],
        [$h_max$], [$7.5 times 10^(-2)$], [$2.4 times 10^(-1)$],
        [$h_"mean"$], [$3.2 times 10^(-2)$], [$7.9 times 10^(-2)$],
        [$|Delta H|$ (drift)], [$2.1 times 10^(-3)$], [$2.8 times 10^(-5)$],
        [*RHS evals*], [$approx$ 9 474], [$approx$ 7 596],
        [*Order*], [$5(4)$], [$8(5)$],
      ),
      caption: [Hard preset: $t=50\,"s"$, $h_0=10^(-3)$, $epsilon_"rel"=10^(-6)$],
    )
  ],
  [
    DOP853 takes *2.5$times$ fewer steps* with *3$times$ larger average $h$* while achieving *100$times$ better energy conservation* ($2.8 times 10^(-5)$ vs $2.1 times 10^(-3)$).

    #v(1em)

    Same tolerance $=>$ DOP853 reaches deeper into chaotic fidelity at *less cost*.
  ],
)

#v(0.5em)

#tblock(title: [The gap widens with tighter tolerance])[
  At $epsilon_"rel" = 10^(-12)$ DOP853 still handles the integration elegantly; RK45 may stall, rejecting nearly every step once the Lyapunov instability sets in.
]

== Adaptive Runge-Kutta RK45 vs DOP853

#figure(
  image("images/adapt.png", width: 67%),
  caption: [Adaptive Runge-Kutta RK45 with relative tolerance of $10^{-6}$, h],
)

== Takeaways

#set list(spacing: 0.8em)
- *Order matters* — global error $cal(O)(h^p)$; RK8 beats RK4 by a factor of $h^4$ at the same cost.
- *Adaptive $h$ saves work* — concentrate evaluations where dynamics are fast, stretch where they are slow.
- *Embedded pairs are free* — same stages, two weight rows; FSAL saves one extra evaluation per accepted step.
- *$Delta H$ is a proxy, not a proof* — correct energy $eq.not$ correct trajectory on a chaotic shell.
- *Use DOP853* — order 8, adaptive, FSAL: the efficient choice for long chaotic integration.

#tblock(title: [Bottom line])[Higher order $+$ adaptive $h$ $+$ embedded error $=$ more accuracy for less compute.]
#tblock(
  title: [The efficient solver],
)[DOP853 adaptive $=$ right order $+$ right step size $+$ free error estimate. Three ideas, one method.]


== Thanks you!!

#grid(
  columns: (1fr, 1fr),
  gutter: 1.4em,
  [
    *References:*

    - #link("https://link.springer.com/book/10.1007/3-540-30666-8")[Geometrical Numerical Integration - Ernst Haire]

    - #link(
        "https://tobydriscoll.net/fnc-julia/ivp/adaptive-rk.html",
      )[Fundamentals of Numerical Integration - Toby Driscoll]
  ],
  [
    #figure(
      image("images/thanks.jpg", width: 3.1in, height: 3.1in),
      caption: [
        #link(
          "https://github.com/jorgemunozl/phisy/tree/master/src/physi/double_pendulum",
        )[#fa-github() Github Repo for Reproducibility]
      ],
      numbering: none,
    )
  ],
)
