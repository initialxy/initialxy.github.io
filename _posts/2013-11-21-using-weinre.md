---
layout: post
author: initialxy
title: "Using Weinre"
description: "Things to watch out for when using Weinre."
category: "Discussion"
tags: [JavaScript, Weinre, Debugging]
---
{% include JB/setup %}

[Weinre](http://people.apache.org/~pmuellr/weinre/docs/latest/) (pronounced like "winery") is a lovely tool to provide JavaScript console and DOM tree inspection for remote devices that don't ship with a remote debugger. I've been using it for quite a while especially when I got issues with Android's legacy WebView. (I say legacy, because the new WebView on Android 4.4 KitKat is now implemented with [Chromium](http://developer.android.com/about/versions/android-4.4.html#Behaviors) and has built-in remote debugger, but the older one doesn't.) I'd like to discuss some things to watch out for when using Weinre as well as some tips. <!--more-->

### Fundamentals

This post assumes you already know the basics of getting Weinre to run. I won't cover anything related to installation of start up. But before you start there's a few very important things you need to remember. I've been using version 2.0.0 at time of this writing.

* Weinre is **not** a JavaScript debugger. For JavaScript debugging, you may be interested in [Aardwolf](http://lexandera.com/aardwolf/), which is a very different piece of art.
* Weinre does not print JavaScript errors. Due to the fact that Weinre is just like every other piece of JavaScript that's running in your environment, it cannot intercept JavaScript Errors such as `SyntaxError` or `ReferenceError`. So if things just stopped working and Weinre console didn't print anything, you might just have ran into an error but Weinre couldn't tell you. It is a good idea to run your code on a desktop browser (if possible) and watch desktop browser's console to cache these errors.

### Issues

* Weinre is sluggish. You could inspect elements and make changes just like Chrome's inspector, but you gotta be patient. Sometimes when you change a CSS property and nothing happens, it might just be Weinre taking its time.
* Weinre's DOM tree does not get refreshed as often as you wish. Most of the time it's outdated following some DOM manipulations. I've found that the best way to force is to refresh is to go back to _Remote_ tab, click on the current target (that's already in green) then go back to _Elements_ tab. Unfortunately you will lose the DOM tree you expanded earlier, but at least you get to see the most up to date data.

![Remote target](/static/images/2013-11-21-using-weinre/remote_refresh.jpg)

* Weinre disconnects easily, especially when your WebView is sent to background or your device goes to sleep. If you react quickly enough, you might be able to save it. If you suspect Weinre has disconnected, you could go to _Console_ tab and run `alert("hello");` to see if there's any reactions.
* Weinre is bounded by the same rules as any other JavaScript, HTML and CSS, so don't be surprised if certain things seems wanky, such as element highlight went under a element of yours that has massive z-index.
* `console.log();` will throw an exception. This is more of a bug than issue, but worth mentioning. So make sure you fully take things out when console debugging, .

### Tips

Aside from the tips I have already mentioned above and some common Chrome tricks like `$0`, I want to note that even though Weinre does not provide a JavaScript debugger, it at least provides a functional JavaScript console that runs in the same environment as your device. That means you can use it to inspect APIs, DOMs and your JavaScript objects. Oftentimes your JavaScript variables and objects are not available to the global scope, but you could try to expose the ones you need in your code to global score by assigning them to a global variable without using the `var` keyword eg. `g = this;`. Then you can inspect them in the console.
