# Python Playground

Python Playground is a free open source [live coding](https://en.wikipedia.org/wiki/Live_coding) environment that lets you explore Python by creating 2D graphics, games and simulations, heavily inspired by [Bret Victor's ideas](http://worrydream.com/#!/LearnableProgramming).

UI is based on cross-platform Python framework [Kivy](https://kivy.org/) that runs on Linux, Windows, OS X, Android, iOS, and Raspberry Pi.

The physics engine utilizes the [Chipmunk2D](https://chipmunk-physics.net/) — fast and portable 2D rigid body physics library written in C.

![Screenshot](https://github.com/dsv57/playground/blob/master/playground/data/images/scr.png)

Please note that Python Playground is under heavy development and is incomplete in various areas.

## Features

* Fast code execution and tracing.
* Live coding feature helps you quickly and easily experiment and fix bugs.
* Smart smooth transitions from previous graphical output.
* Code autocompletion using [Jedi](https://jedi.readthedocs.io/en/latest/).
* Nice colors using perceptually modeled [CIECAM16 color model](https://onlinelibrary.wiley.com/doi/abs/10.1002/col.22131) and alpha blending in linear space.
* LOGO Turtle mode (in progress).
* Sokoban mode (in progess).
* Game development and rigid body physics simulation.


## Building / Setup / Installation

Prerequisites:
* kivy,
* pymunk,
* scipy.

```
pip -r requirements.txt
python3 -m playground
```


## Future work

* Think about new vector engine ([AGG](https://en.wikipedia.org/wiki/Anti-Grain_Geometry), [Skia](https://skia.org/), [Pathfinder](https://github.com/servo/pathfinder) or plain ugly Kivy/OpenGL).
* Add text labels.
* Graphic object selection and properties panel.
* Multithreaded & fast ([PyPy](https://www.pypy.org/)) execution.
* Use simple and fast [Oklab color model](https://bottosson.github.io/posts/oklab/) (add saturation scale) and drop patched [colorio](https://github.com/nschloe/colorio) dependency.
* More featureful code editor like that of [Codea](https://codea.io/).
* Collaboration & teamwork.
* Fix Sokoban and Turtle modules.
* Add geometric constraints like in [Apparatus hybrid graphics editor](http://aprt.us/).
* Symbolic mathematics and geometry using [SymPy](https://docs.sympy.org/latest/index.html).


## Related work

[![Bret Victor — Inventing on Principle](https://github.com/dsv57/playground/blob/master/playground/data/images/inventing-on-principle.jpg)](https://vimeo.com/38272912)

* [Swift Playgrounds](https://www.apple.com/swift/playgrounds/) is a app for iPad and Mac that makes it fun to learn and experiment with Swift.
* [Algodoo](http://www.algodoo.com/) is a physics-based 2D sandbox.
* [Codea](https://codea.io/) for iPad lets you create games and simulations in Lua — or any visual idea you have. 
* [Kojo](https://www.kogics.net/kojo) is a programming language and IDE for computer programming and learning that has many different features that enable playing, exploring, creating, and learning in the areas of computer programming, mental skills, (interactive) math, graphics, games and other. Kojo draws ideas from the programming languages Logo and Processing.
* [The Programmer's Learning Machine (JavaPLM)](http://people.irisa.fr/Martin.Quinson/Teaching/PLM/index.html) is a free cross-platform programming exerciser. It lets you explore various concepts of programming through interactive challenges, that you can solve in either Java, Python or Scala (support for the C language is experimental).
* [Apparatus](http://aprt.us/) is a hybrid graphics editor and programming environment for creating interactive diagrams.

See also:
* [List of educational programming languages](https://en.wikipedia.org/wiki/List_of_educational_programming_languages).
* [Live coding](https://en.wikipedia.org/wiki/Live_coding), [curated list](https://github.com/toplap/awesome-livecoding).
* Powerful interactive graphics frameworks [Paper.js](http://paperjs.org/) and [Flutter](https://docs.flutter.dev/).

## Licence

This code is under a GPLv3 License.

[Shader-Based Antialiased Polylines](https://jcgt.org/published/0002/02/08/) (C) 2013 Nicolas P. Rougier.
