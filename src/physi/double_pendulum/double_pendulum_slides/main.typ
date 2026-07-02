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
      caption: "Double Pendulum",
    )
  ],
)

== From Taylor series to Runge Kutta
Instead of the use Euler:
$
  y_(n+1) = y_n + h f(y_n, t_n)
$
We popose:
$
  y_(n+1) = y_n + h (a_1 k_1 + a_2 k_2)
$

Develop the Taylor series expansion to obtain the coefficients.

Mid point, middle points and

== Working examples with  RK4 and RK8

To obtain the coefficients you would need to develop a third order Taylor series expansion.

You use the Butcher tableau to obtain the coefficients, or the Butcher three.

For RK4 and RK8.

== Adaptive step-size control

Adaptive step-size control.

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
