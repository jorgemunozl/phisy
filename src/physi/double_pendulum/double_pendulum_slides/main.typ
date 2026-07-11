#import "@preview/cetz:0.4.2"
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

Using this motivation Runge and Kutta propose the follow form to obtain the solution.
$ k_i = f(t_n + c_i h, y_n + h sum_j a_(i j) k_j) and y_(n+1) = y_n + h sum_i b_i k_i $

_How do you obtain coefficients $(a_(i j) , b_i, c_i)$?_

#grid(
  columns: (2fr, 0.7fr),
  gutter: 1em,
  [
    $ sum_(j=1) a_(i j) = c_i $
    Using the *Taylor Polinomial* expansion of $y(t+h)$, $y'=f(t, y)$.

    Therefore you asure:
    $ y_1 - y(t_0 + h) = cal(O)(h^(p+1)) text("as") h -> 0 $

  ],
  [
    #figure(
      image("images/table.png", width: 120%),
      caption: [RK coefficients],
    )
  ],
)

== Runge Kutta for the Double Pendulum

#let bf(x) = math.bold(math.upright(x))

We have second derivatives; use vectors instead of scalars.

$ bf(u)=(theta_1, dot(theta_1), theta_2, dot(theta_2)) $

$ dot(bf(u))=(dot(theta_1), dot.double(theta_1), dot(theta_2), dot.double(theta_2)) $

The update would be:

$ bf(u_(n+1)) = bf(u_n) + h sum_i b_i bf(k_i) $

== Working example with  RK4 and RK8

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #figure(
      image("images/rk4_t.png", width: 80%),
      caption: [Tableau for RK4],
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

    #cetz-canvas({
      import cetz.draw: *

      // Axes
      let W = 5.5
      let H = 3.2
      line((0, 0), (W, 0), mark: (end: ">"), name: "xax")
      line((0, 0), (0, H), mark: (end: ">"), name: "yax")
      content((W + 0.3, 0), $t$)
      content((0.05, H + 0.25), $Delta H$)
      content((0, -0.25), $0$)

      // Euler — blows up (steep exponential-like)
      bezier(
        (0, 0.05),
        (W * 0.55, H * 0.95),
        (W * 0.15, 0.15),
        (W * 0.4, H * 0.7),
        stroke: (paint: red, thickness: 1.5pt),
      )
      content((W * 0.57, H * 0.95 + 0.22), text(fill: red, size: 8pt)[Euler])

      // RK4 — gentle polynomial drift
      bezier(
        (0, 0.02),
        (W * 0.92, H * 0.48),
        (W * 0.35, 0.08),
        (W * 0.7, H * 0.3),
        stroke: (paint: blue, thickness: 1.5pt),
      )
      content((W * 0.94, H * 0.48 + 0.22), text(fill: blue, size: 8pt)[RK4])

      // RK8 — near flat
      bezier(
        (0, 0.01),
        (W * 0.92, H * 0.06),
        (W * 0.4, 0.02),
        (W * 0.7, 0.055),
        stroke: (paint: green, thickness: 1.5pt),
      )
      content((W * 0.94, H * 0.06 + 0.22), text(fill: green, size: 8pt)[RK8])
    })
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

    - Energy drift *confirms method order* without needing a reference solution.
    - But: correct energy $≠$ correct trajectory — a chaotic orbit can wander the right energy shell in the *wrong direction*.
  ],
)

== Working example with RK4 and RK8

#figure(
  image("images/video.png", width: 70%),
  caption: [Comparison of RK4 and RK8 solutions with $h=0.001$],
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
    #cetz-canvas({
      import cetz.draw: *
      let W = 5.0
      let Hc = 2.6

      // Axes
      line((0, -0.5), (W, -0.5), mark: (end: ">"))
      line((0, -0.5), (0, Hc), mark: (end: ">"))
      content((W + 0.28, -0.5), text(size: 9pt)[$t$])
      content((-0.28, Hc + 0.05), text(size: 9pt)[$y$])

      // Curve: smooth — dynamic burst — smooth
      bezier(
        (0, 0.9),
        (2.0, 1.05),
        (0.5, 0.7),
        (1.5, 1.3),
        stroke: (paint: black, thickness: 1.3pt),
      )
      bezier(
        (2.0, 1.05),
        (3.2, 0.95),
        (2.35, 2.3),
        (2.9, -0.15),
        stroke: (paint: black, thickness: 1.3pt),
      )
      bezier(
        (3.2, 0.95),
        (W, 1.1),
        (3.7, 0.65),
        (4.4, 1.45),
        stroke: (paint: black, thickness: 1.3pt),
      )

      // Step markers: wide — narrow — wide
      let marks = (0.0, 0.85, 1.7, 2.1, 2.32, 2.5, 2.65, 2.78, 2.9, 3.4, 4.15, 5.0)
      for x in marks {
        line((x, -0.62), (x, -0.38), stroke: (thickness: 1pt))
      }

      // Region labels below axis
      content((0.85, -0.9), text(size: 7.5pt)[large $h$])
      content((2.5, -0.9), text(size: 7.5pt)[small $h$])
      content((4.1, -0.9), text(size: 7.5pt)[large $h$])

      // Region labels on the curve
      content((0.85, 2.1), text(size: 7.5pt, fill: gray)[smooth])
      content((2.5, 2.4), text(size: 7.5pt, fill: red)[dynamic])
      content((4.1, 2.1), text(size: 7.5pt, fill: gray)[smooth])
    })

    #v(0.4em)
    The solver *concentrates* evaluations where dynamics are fast, and stretches $h$ where they are slow.
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
    #cetz-canvas({
      import cetz.draw: *
      let aw = 2.8 // A block width
      let ah = 1.8 // A block height
      let bh = 0.42 // b-row height
      let cw = 0.52 // c column width

      // c block
      rect((-cw, 0), (0, -ah), fill: luma(245), stroke: (paint: luma(160), thickness: 0.5pt))
      content((-cw * 0.5, -ah * 0.5), text(size: 9pt)[$bold(c)$])

      // A block (lower triangular, gray)
      rect((0, 0), (aw, -ah), fill: luma(232), stroke: (paint: luma(160), thickness: 0.5pt))
      // Triangle outline inside A to hint at lower-triangular structure
      line((0, 0), (0, -ah), stroke: (paint: luma(150), thickness: 0.4pt))
      line((0, -ah), (aw, -ah), stroke: (paint: luma(150), thickness: 0.4pt))
      line((0, 0), (aw, -ah), stroke: (paint: luma(150), dash: "dashed", thickness: 0.6pt))
      content((aw * 0.35, -ah * 0.5), text(size: 9pt)[$A$])
      content((aw * 0.78, -ah * 0.18), text(size: 7pt, fill: luma(140))[0])

      // Thick separator between A and b rows
      line((-cw, -ah), (aw, -ah), stroke: (paint: black, thickness: 1.0pt))
      // Vertical separator between c/0 and A/b
      line((0, 0.15), (0, -ah - 2 * bh), stroke: (paint: luma(120), thickness: 0.5pt))

      // b row — order p (blue)
      rect((-cw, -ah), (0, -ah - bh), fill: luma(245), stroke: (paint: luma(160), thickness: 0.4pt))
      rect((0, -ah), (aw, -ah - bh), fill: rgb("#ddeeff"), stroke: (paint: blue, thickness: 0.5pt))
      content((aw * 0.5, -ah - bh * 0.5), text(size: 9pt, fill: blue)[$bold(b)^top quad$ order $p$])

      // b-hat row — order p+1 (red)
      rect((-cw, -ah - bh), (0, -ah - 2 * bh), fill: luma(245), stroke: (paint: luma(160), thickness: 0.4pt))
      rect((0, -ah - bh), (aw, -ah - 2 * bh), fill: rgb("#ffdddd"), stroke: (paint: red, thickness: 0.5pt))
      content((aw * 0.5, -ah - bh * 1.5), text(size: 9pt, fill: red)[$hat(bold(b))^top$ order $p+1$])
    })

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

== Testing out Adaptive Runge-Kutta RK45 vs DOP853

// Here goes table

== Adaptive Runge-Kutta RK45 vs DOP853

#figure(
  image("images/dp.webp"),
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
