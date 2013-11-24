---
layout: post
author: initialxy
title: "Minimal Modularized Web Components Demo"
description: "Minimal Modularized Web Components Demo and discussions on the brave new technology."
category: "Demo"
tags: [JavaScript, HTML5, Web Components, Demo]
---
{% include JB/setup %}

A few months ago I came across the following video from this year's Google I/O. Long story short, Web Components are browser vendor's way of implementing GUI widgets, except now they are trying to make their API publically accessible and standardize into HTML5. I became extremely excited, because it could effectively replace all of the existing JavaScript GUI frameworks out there with an elegant, organized, encapsulated fashion supported natively. At this point I was heavily involved in development with [Dojo](http://dojotoolkit.org/) and its way of making [widgets](http://dojotoolkit.org/reference-guide/1.9/quickstart/writingWidgets.html). Perhaps a bit too much. (I literaly have Dojo 1.9 source code setup as a project in my [Sublime Text 2](http://www.sublimetext.com/2), and studied it extensively.) I have a bit of love-hate relationship with Dojo, and I wanted to see a native way of implementing widgets.

[![Web Components: A Tectonic Shift for Web Development](http://img.youtube.com/vi/fqULJBBEVQE/0.jpg)](http://www.youtube.com/watch?v=fqULJBBEVQE)

Web Components were so cutting edge at the time (actually even at the time of this writing) that you literally have to download the nightly build of [Chromium](http://www.chromium.org/), enable bunch of experimental features explicitly, and you still only have a patchy support of it. In this post, I'd like to discuss the current state of Web Components, and show you a mininal modularized demo. I say modularized, I mean breaking it down into as many files as possible.<!--more-->

First thing first, let's just get straight to the point. You can see my demo [here](/static/files/2013-11-22-minimal-modularized-web-components-demo/demo/). Again, you need to have the latest version of [Chromium](http://www.chromium.org/). (At time of this wirting, the latest Chromium is 33 and Chrome is 31.) Open _chrome://flags_ in address bar and enable:

* Enable Experimental JavaScript
* Enable experimental Web Platform features
* Enable HTML Imports

You can also find awesome resources covering Web Compoments features from [HTML5Rocks](http://www.html5rocks.com/en/):

* [HTML template](http://www.html5rocks.com/en/tutorials/webcomponents/template/)
* [Shadow DOM](http://www.html5rocks.com/en/tutorials/webcomponents/shadowdom/)
* [Custom element](http://www.html5rocks.com/en/tutorials/webcomponents/customelements/)
* [HTML import](http://www.html5rocks.com/en/tutorials/webcomponents/imports/)

Now you know what Web Components is all about. Aren't you excited? I actually realized that some of these concepts are somewhat analogical to Dojo's [widgets](http://dojotoolkit.org/reference-guide/1.9/quickstart/writingWidgets.html) implementation, which I found fascinating. (great minds think alike eh?)

Like I mentioned earlier, I modularized my demo as much as I could, that means I wanted each aspect of implementation to be done in its own separate file, and then import into one. So my import structure looks like this:

    index.html <-- cus-widget.html <----- cus-widget.js
                                      +-- cus-widget.css

The idea is that all index.html has to worry about is to import _cus-widget.html_ then it get simply use the `<cus-widget>hello<cus-widget/>` tag to its advantage. _cus-widget.html_ implements this rather simple custom widget and it impports JavaScript and CSS from separate files. Unfortunately this time around, I actually had less success than a few of months ago. First of all, [Polymer](http://www.polymer-project.org/) is a project that's intended to provide polyfill for Web Components. Few months ago, I _had_ to use Polymer because event the latest nightly build of Chromium didn't have all of the Web Components features (namely HTML import was missing), and I actually managed to get it to work somewhat close to idea setup. Today, I got the latest version of Polymer (0.0.20131107) and I immediately hit a [bug](https://github.com/Polymer/polymer/issues/290). But that's ok, I easily worked around it. But now I noticed that the styling for my custom widget was missing. Now ideally, I should be able to import CSS as part of the template, as per this [HTML5Rocks article](http://www.html5rocks.com/en/tutorials/webcomponents/imports/):

    <template>
        <link rel="stylesheet" href="polymer-ui-tabs.css">
        <polymer-flex-layout></polymer-flex-layout>
        <shadow></shadow>
    </template>

This should ensure an isolated CSS name space for the custom widget. So I originally had this in my _cus-widget.html_:

    <template id="cus-widget_template">
        <link rel="stylesheet" href="cus-widget.css"></link>
        <div>
            <content></content>
            This is a custom widget!
        </div>
    </template>

My _cus-widget.css_ looked like this:

    div {
        border: 1px solid red;
        padding: 1em;
        display: block;
    }

This was all working before, and now it stopped working. So I scrolled down some more on the same HTML5Rocks article to find some clues. Appearently I could grab my CSS import from my _cus-widget.html_ and dump it into the main document, _index.html_. I had to add the following to my _cus-widget.js_:

    var styleSheet = document.currentScript.ownerDocument
            .querySelector("link[rel='stylesheet'][href$='cus-widget.css']");
    document.head.appendChild(styleSheet.cloneNode(true));

At this point I hit yet another weird Polymer bug. So I removed Polymer from _index.html_. (In case you are wondering if my original code would have worked without Polymer, the answer is no, it still didn't import my style sheet on Chromium 33.) But then the above piece of code failed to find my style sheet. So I had to move the CSS import out of `<tempate>` tag and into `<head>` tag like so:

    <head>
        <title>CusWidget</title>
        <link rel="stylesheet" href="cus-widget.css"></link>
    </head>

Finally my CSS is imported (as evident on the _Network_ tab of Chromium's debugger). But I still don't see my style. I figured at this point, it's just a main document style sheet, so I gotta style my _cus-widget_ like this:

    cus-widget {
        border: 1px solid red;
        padding: 1em;
        display: block;
    }

Yay! It's finally showing everything correctly, or is it? There's definitely something fundamentally **wrong** with this demo if you haven't noticed. Now my CSS is just a main document style sheet like every other imported CSS, which completely defeats the purpose of having CSS for Shadow DOMs. There's no longer an isolated name space for _cus-widget.css_, and as you can see earlier, it can no longer select and style Shadown DOMs.

In conclusion, Web Components is very promising, but it's most defintely still too immature for prime time. I shall wait patiently for it to become production-ready.
