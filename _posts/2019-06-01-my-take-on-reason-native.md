---
layout: post
author: initialxy
title: "My Take on Reason Native"
description: "My thoughts on Reason Native after playing with it"
category: "Discussion"
tags: [OCaml, JavaScript, Native, Reason]
---
{% include JB/setup %}

I recently became interested in [Reason](https://reasonml.github.io/), which is an alternative syntax for [OCaml](https://ocaml.org/) in an attempt to keep the JavaScript folks more comfortable writing a "mostly pure" functional language. Judging by its own site, they seem to put a lot of focus on developing Web front end in conjunction with [React](https://reasonml.github.io/reason-react/). As someone who works professionally with [Haskell](https://www.haskell.org/) and JavaScript (among other things), but no prior knowledge of OCaml, I feel I'm a perfect candidate to dive into Reason and get a feel of it. However I did not use it to develop a web app, instead, I chose to use it for [native development](https://reasonml.github.io/docs/en/native). I feel this is the best way to consume the language itself instead of being heavily influenced by React and JavaScript interop. So I wrote a personal [Raspberry Pi project](https://github.com/initialxy/initialxy-frontpoint-scheduler) with Reason Native. Just to clarify, this post is meant to be an opinion piece rather than an in-depth review of Reason as a language.<!--more-->

### What is Reason?
Reason is not really a new language, but an alternative syntax of OCaml. You can read its official explanation [here](https://reasonml.github.io/docs/en/what-and-why). Since this is an opinion piece, my take on it (which may or may not be aligned with its author's intentions) is that Reason along with Reason React are meant to be an alternative to [elm](https://elm-lang.org/), which is another functional language that's meant to develop web apps. However, instead of creating a new language, Reason hopes to accelerate its adoption by piggybacking on the OCaml ecosystem as well as its community while appealing to web developers by being not too overwhelming to someone already familiar with JavaScript. Why functional language? As someone who works with Haskell, I can testify that a good, modern functional language with little to no side effects and a strong type system can save a lot of errors during compile time. I have personally experienced the euphoria of "when it compiles, it works" when writing Haskell. This is not something even a well-typed JavaScript alternative like [TypeScript](https://www.typescriptlang.org/) can achieve, as JavaScript is full of side effects. As someone who pulled plenty of my hairs when learning Haskell, I can understand that learning a "mostly pure" functional language can be overwhelming to someone coming from a more traditional imperative language. So I believe this is the gap Reason is trying to close. However, despite my understanding that Reason is a more web-focused language, I decided to evaluate its native application development experience instead. So this post does not apply to Reason's web app development experience.

### Project Objectives
My project is meant to a simple scheduler that interacts with a web API. It is meant to run on Raspberry Pi, which uses an ARM processor. I won't get too deep into what my project is meant to do. I will just cover some criteria that I wanted to evaluate.
* Setup and running a Hello World! project.
* Learning the language itself.
* Echosystem.
* Community support.

More specifically, I also wanted to accomplish integration with some technologies that I consider to be essential.
* Advanced HTTP request and response handling.
* JSON parsing.
* SQLite3.
* Command line parsing.
* Asynchronous programming.

I'm happy to report that I managed to accomplish my goals. But how was the experience.

### Hello World!
Setting up the Hello World! project was quite easy. Following Reason's [guide](https://reasonml.github.io/docs/en/quickstart-ocaml), I installed [esy](https://esy.sh/) and checked out their [bootstrap repo](https://github.com/esy-ocaml/hello-reason). From there, it compiled and ran just fine on my Arch Linux as well as Mac OS X. I stripped down the bootstrap a bit to get to a cleaner state. However, when I deployed my project onto Raspberry Pi, I immediately ran into problems. Esy doesn't seem to work on ARM processor despite it installs just successfully. I still managed to compile my project with [dune](https://dune.build/), but the experience was quite rocky. I ran into a lot of trouble just to find a version of OCaml that met the minimum and maximum versions of all of the third-party packages that I installed. Even when I finally installed all of the packages and got it to compile and run, [cohttp](https://github.com/mirage/ocaml-cohttp), which was the HTTP client I used, refused to connect to HTTPS. I managed to figure out that in addition to needing [lwt_ssl](https://github.com/ocsigen/lwt_ssl), it also needed to be compiled with [tls](https://opam.ocaml.org/packages/tls/), which was not an issue I observed on Arch Linux nor Mac OS X. However, in the end, I prevailed. My project is now running successfully on Raspberry Pi.

### Learning Reason
Learning Reason itself was quite easy. As someone who is already fluent in Haskell and JavaScript, its [tutorial page](https://reasonml.github.io/docs/en/overview) made a lot of sense. I managed to follow along easily. In fact, I feel it did not cover enough of the language. As I finished reading its tutorial, I was still left wondering how its monadic expression, type class, and function composition would look like. But that was enough to dive into the language. I decided to just learn as I play with it. However as I dug deeper and deeper, I came to the realization that the rest of the language is literally just OCaml in its naked form. For example `>>=` works just fine for monadic infix operator. OCaml does not have dot composition so neither does Reason. Reason has `->` and `|>` for piping instead. I became more and more familiar with OCaml instead as I moved forward. Though it was easy for me because I'm already fluent with Haskell. But I feel it defeats the purpose of having an alternate syntax to keep JavaScript folks comfy if I ended up learning OCaml anyways. So assume you are already familiar with OCaml, here is how I'd summarize Reason:
1. Use JavaScript/TypeScript like lambda syntax for function declaration. Recursive function needs `rec` keyword. eg.
    ```javascript
    let foo = (param1: string, param2: float, param3: list(int)): int => { ... }
    let rec fooRec = param => if (param > 0) {
      fooRec(param - 1);
    } else {
      "Hello World!";
    }
    ``` 
2. Use parentheses for function evaluation. Partial function and currying work equivalently. eg.
    ```javascript
    foo(arg1, arg2, arg3);
    let partial = foo(arg1, arg2);
    partial(arg3);
    ```
3. Pattern matching almost always needs `switch` statement. eg.
    ```javascript
    let foo = param => switch(param) {
      | 0 => "It's zero"
      | 1 => "It's one"
      | _ => "I can't math"
    }
    ```
4. Blocks use braces. eg.
    ```javascript
    let foo = param => {
      let a = 1;
      let b = 2;
      let c = 3;
      a + b + c;
    }
    ```
5. Semi-columns are optional, but [refmt](https://github.com/reasonml/reason-cli) likes to put it all over the place. 

I really can't think anything else off the top of my head, which begs the question: why didn't I just write OCaml? The amount of comfort Reason brought was quite minimal if I was not already familiar with Haskell.

### Ecosystem
Reason uses esy as its package manager, but you can also pull packages from [opam](https://opam.ocaml.org/) to take advantage of OCaml's ecosystem. Looking for the right package was relatively easy, and as far as I have experienced, there was always a package that can satisfy my needs. However, I found OCaml's (hence Reason's) standard library to be lacking. Certain trivial operations often required third-party libraries, such as calendar operations and getting an absolute path of a relative path. These things are easily achievable in Java or Python with their standard libraries. While documentation can often be found, figuring out how to use them can often be challenging, which leads us to community support.

### Community Support
A healthy language often requires a vibrant community. When you are developing Java, Python or JavaScript, you can often Google the exact problem you are having and quickly find a solution or sample code somewhere on the internet. Unfortunately, this was not the case for OCaml and even less so for Reason. Even trivial problem like getting a timestamp of a specific time of day has literally no explicit answer nor samples. I often had to just dig through pages of documentation to figure it out. A modern functional language such as Haskell can often express its intention by its types, hence one could find the right function by just searching its types. This was still a huge obstacle to clear if you don't know what to look for. Even figuring out how to use some trivial operations of popular tooling required weeding through pages of documentation. I wanted to add another source file to be compiled by dune, but Google found no sample of how to accomplish it. Turns out you needed to add the `modules` strata into dune file. When I was trying to figure out how to use SQLite3, its [core library](https://github.com/mmottl/sqlite3-ocaml) was easy enough, but it didn't have SQL escape capability, which was concerning. So I decided to use [caqti](https://github.com/paurkedal/ocaml-caqti). I had to search GitHub to find other open source projects that use it in order to serve as samples. I don't recall it being this difficult to find out something like this in any other languages that I used. Furthermore, documentation and samples of OCaml libraries are of course written for OCaml. I had to pick up significant amount of OCaml in order to digest them. This again, begs the question: why didn't I just learn and write OCaml in the first place?

### Technologies
I tried my hand on some technologies that I believe is quite essential. I used [cohttp](https://github.com/mirage/ocaml-cohttp) for HTTP client, [caqti](https://github.com/paurkedal/ocaml-caqti) with [SQLite3](https://github.com/mmottl/sqlite3-ocaml) for storage, [lwt](https://github.com/ocsigen/lwt) for asynchronous programming, [yojson](https://github.com/ocaml-community/yojson) for JSON parser and the command line parser was built-in. I managed to get all of these working, but as I noted in the above section, they often lacked good samples and community support, which meant I spent hours just trying to figure out how to use them. That being said, I'm impressed by Reason's integration with Lwt, though to be fair, at this point, OCaml syntax applies. Monadic expression works well and even [lwt_ppx](https://ocsigen.org/lwt/3.2.1/api/Ppx_lwt) works great. I'm impressed by the fact that OCaml allows syntax extension by syntax tree rewriters. I'm even more impressed that it works with Reason without a hitch. The way I understand is that `lwt` provides [Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise) while `lwt_ppx` provides [Async/Await](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function). The additional syntax is provided by a syntax rewriter, namely `lwt_ppx`. It works very much analogical to Async/Await, though with some differences. Notably, `lwt_ppx` does not have async function, so return values are not implicitly wrapped with `lwt.t`. You have to remember to wrap it with `return`, though the compiler will complain if you didn't. eg.

```javascript
let asyncFoo = (param: string): Lwt.t(int) => { ... }
let asyncBar = (param: string): Lwt.t(int) => {
  let%lwt res = asyncFoo(param);
  return(res);
}
// or just return the lwt.t result directly
let asyncBar = (param: string): Lwt.t(int) => asyncFoo(param);
// this is not ok
let asyncBar = (param: string): Lwt.t(int) => {
  let%lwt res = asyncFoo(param);
  res;
}
```

### Conclusion
While I had a lot of fun writing Reason, I feel I came out of the other end learning more about OCaml. Like I already mentioned above, I'm questioning why I didn't just learn and code in OCaml in the first place. So my final take on Reaon Native is that it has very little purpose if you want to develop a native application. That is not to say that it has diminishing value on the web. Like I mentioned in the beginning, this post applies only to Reason Native and does not apply to Reason React. I feel it might be a formidable competitor to elm, but I have yet to try it out on my own.