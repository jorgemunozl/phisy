#import "@preview/cetz:0.4.2"
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
    title: [Numerical Methods for the double pendulum],
    short-title: [],
    subtitle: [Runge-Kutta, Adaptive, High Order],
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
      x_2 = L_1 sin theta_1 + L_2 sin theta_2 and y_2 = - L_1 cos theta_1 - L_2 cos theta_2
    $
    $ cal(L) = T - U $
    Using the Euler Lagrange equations you obtain:

    $
      dot.double(theta)_1 = f_1(theta_1, theta_2, dot(theta)_1, dot(theta)_2) \
      dot.double(theta)_2 = f_2(theta_1, theta_2, dot(theta)_1, dot(theta)_2)
    $
  ],
  [
    #figure(
      image("images/dp.webp"),
      caption: [Double Pendulum],
    )
  ],
)

== Motivatig Kutta

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

Energy as a way to compare

Global error

== Working example with RK4 and RK8

#figure(
  image("images/video.png", width: 70%),
  caption: [Comparison of RK4 and RK8 solutions],
)

== Adaptive step-size control

Error for a order method

$ p -> u_(i+1) $

$ p+1 -> tilde(u)_(i+1) $

$ E_i(h) = \|u_i - tilde(u)_(i+1)\| $

$E_i(h)<epsilon$

$E_i(h)approx C h^(p+1)$

$E_i(q h) approx q^(p+1) C h^(p+1) approx epsilon$

$q approx (epsilon/E_i(h))^(1/(p+1))$

== Embedding Runge Kutta



== Reference Solution Strategy

Reference solution strategy.

== Energy Drift Comparison

Variation of the energy drift with respect to the step size.

== Sensitive dependence & Lyapunov exponent

Sensitive dependence on initial conditions.

== Conclusion:

Some ideas

// No idea
== Pendents

#grid(
  columns: (auto, 1fr),
  gutter: 0.5em,

  // Unchecked checkbox + task
  box(stroke: 1pt, width: 0.7em, height: 0.7em), [*Task 1:* Evaluate the model with the remaining datasets],

  box(stroke: 1pt, width: 0.7em, height: 0.7em), [*Task 2:* Include LiBF#sub[4] and BLi#sub[3] data in training],

  box(stroke: 1pt, width: 0.7em, height: 0.7em), [*Task 3:* Repeat for MACE-MP-0 and compare results],
)
