---
layout: post
author: initialxy
title: "Troubles with device-width"
description: "device-width and its inconsistent behaviours on different mobile browsers."
category: "Discussion"
tags: [CSS3, HTML5, Mobile Web, Media Query]
---
{% include JB/setup %}

Mobile web developers are probably aware of what [media queries](https://developer.mozilla.org/en-US/docs/Web/Guide/CSS/Media_queries) are. They are extremely useful tools to select resource as well as provide layout tweaks for responsive web designs. One of the well known media query feature is `device-width` along with its variations (`min-device-width`, `max-device-width`). However I have discovered that it has vastly inconsistent behaviours on different mobile browsers. In this article, I'd like to discuss what exactly `device-width` means, its issues as well as solutions. <!--more-->

## What is device-width?

Let's begin by figuring what `device-width` really means. According to [Mozilla's definition](https://developer.mozilla.org/en-US/docs/Web/Guide/CSS/Media_queries#device-width):

> Describes the width of the output device (meaning the entire screen or page, rather than just the rendering area, such as the document window).

This is a very straightforward definition. What it means is that device-width is the screen width. On the other hand, `width` is the window width. The following screenshot makes illustrates this concept.

![width vs device-width](/static/images/2013-11-25-troubles-with-device-width/device-width.jpg)

But what difference does this make on a mobile device? Mobile apps are always in fullscreen anyways. Shouldn't `width` and `device-width` be the same? Not quite. Let's see how it works on different mobile browsers. Before we begin, let's review a very important concept; `px` is not pixel. You've probably heard this many times already. What it means is that the CSS unit `px` does **NOT** translate to screen pixel. A very good example is to look at the transition from [iPhone 3GS](http://en.wikipedia.org/wiki/IPhone_3GS) to [iPhone 4](http://en.wikipedia.org/wiki/IPhone_4). iPhone 3GS has 320x480 screen resolution, while iPhone 4 has 640x960 screen resolution and Apple called it the Retina display. Notice that screen resolution doubled on both dimensions. This yields a problem with its mobile browser. If I made a web site for iPhone 3GS that has width of `320px`, then it would now become tiny and occupy only a half of the width on iPhone 4, which would be very unpleasant. So Apple solved this by defining iPhone 4's `px` resolution to be 320x480, which gave rise to another very important media query feature, `-webkit-device-pixel-ratio`. This value is accessible in JavaScript as `window.devicePixelRatio`. iPhone 4's pixel ratio is set to 2, which makes sense since resolutions were doubled. So `px` is scaled by `-webkit-device-pixel-ratio` in order to translate to actual screen resolution. This is similar to Android's density-independent pixel AKA `dip` or `dp`. Now let's go back to the earlier question. iOS decided that both `width` and `device-width` would be scaled by pixel ratio. So on both the iPhone 4 as well as iPhone 3GS, both of the following media queries will activate in **portrait** mode. You should see a blue background because the second `background-color` overrode the first one.

```css
@media (max-device-width: 320px) {
    body {
        background-color: red;
    }
}

@media (max-width: 320px) {
    body {
        background-color: blue;
    }
}
```

Notice that I said portrait mode. This is where fun begins. On iOS, `device-width` always references to the width of the device in portrait mode, or the way I'd like to put it; always the smaller dimension. (This is equivalent to Android's `sw` resource selector.) While `width` is always the current width of either orientations. So take the above example, and rotate your device to landscape mode, you should see a red background for both iPhone 4 and iPhone 3GS, because now `width` is `480px` while `device-width` is `320px`. Since we are looking for "at most 320px", the second media query will fail.

## Here Comes Trouble

Now let's test this out on Android. Please keep in mind that Android now has two different WebView implementations: a newer implementation [based on Chromium and is available on Android 4.4 KiKat](http://developer.android.com/about/versions/android-4.4.html#Behaviors) and above, and a legacy implementation on Android 4.3 Jelly Beans and below. The [Android Chrome Browser](https://play.google.com/store/apps/details?id=com.android.chrome&hl=en), which could be installed from Google Play on older vesions of Android, should be very similar to the Chromium based WebView. On the legacy WebView, you should notice that `device-width` is **NOT** scaled by pixel ratio, even though `window.devicePixelRatio` could be larger than 1 and `px` is scaled everywhere else in CSS. It's just the `device-width` media query that's "broken". (I have "broken" in quotes, because this behaviour is not really standardized. It's still up to everyone's own definition. We just hoped that Android would follow iOS's suite.) However `width` is still scaled by pixel ratio, and `device-width` references to the smaller dimension. This is a huge deal breaker for usage of `device-width`. Say you have the following media query intended to match tablet.

```css
@media (min-device-width: 768px) {
    /* Styles for tablet. */
}
```

This media query would match all of the iPad models, but not iPhone models (as expected). But, it would also match Nexus 4, which is a phone, because Nexus 4 has screen resolution of 768x1280. This yields `device-width` unusable for device type detection. To make matter more complicated than it already is, Android's Chrome browser as well as the new Chromium based WebView on KitKat behaves in the same fashion as iOS. This is good news, in the sense that looking forward, it is much more consistent. But it also means that there are two sets of significantly different `device-width` behaviours on the same version of OS. Not even a simple OS detection can save you.

Before we move on, let's take a look at between pixel ratio and screen densities.

| pixel ratio | iOS     | Android |
|-------------|---------|---------|
| 0.75        | N/A     | ldpi    |
| 1           | legacy  | mdpi    |
| 1.5         | N/A     | hdpi    |
| 2           | retina  | xhdpi   |
| 3           | N/A     | xxhdpi  |

Up until now, pixel ratio has been rather nice numbers consistently. So you can use `-webkit-device-pixel-ratio` to match them. Even [Android's official documentation](http://developer.android.com/guide/webapps/targeting.html) suggests so. Then we have BlackBerry 10. BlackBerry Z10's pixel ratio is 2.24, with screen resolution of 768x1280 and `px` resolution of 342x571. BlackBerry Q10's pixel ratio is 2.08 with screen resolution of 720x720 and `px` resolution of 346x346. This means you can not use `-webkit-device-pixel-ratio` to match their pixel ratios, instead, you will have to use either `-webkit-max-device-pixel-ratio` or `-webkit-min-device-pixel-ratio`. To make matters worse, BlackBerry 10's `device-width` is not always the smaller dimension, it is simply the current width (similar to `width`).

## Solutions

Given the vast inconsistencies of `device-width`, how are we going detect device types? The answer is to **NOT** use `device-width` and use `width` instead. Notice that `width` has much more consistent behaviour across different platforms. Let's take a look at how [Twitter Bootstrap deals with device type detection](http://getbootstrap.com/css/#grid).

```css
/* Followings are copied and pasted from Twitter Bootstrap. */

/* Extra small devices (phones, less than 768px) */
/* No media query since this is the default in Bootstrap */

/* Small devices (tablets, 768px and up) */
@media (min-width: @screen-sm-min) { ... }

/* Medium devices (desktops, 992px and up) */
@media (min-width: @screen-md-min) { ... }

/* Large devices (large desktops, 1200px and up) */
@media (min-width: @screen-lg-min) { ... }
```

`width` based queries could also help present your layout on desktop browsers, because when a user resizes his/her window to a smaller size, your layout could change to adapt, providing better experiences. You may want to ask; since `width` depends on orientation, how are we going to deal with device type detection? I'd suggest that if you have enough horizontal real estate for a tablet-like layout, might as well let users use it. So the above is fine. But if you **REALLY** want to distinguish between device types, you could combine `width` with `orientation`. eg.

```css
@media (min-width: 768px) and (orientation: portrait),
        (min-width: 1024px) and (orientation: landscape){
    /* Styles for tablet. */
}
```

**TL;DR**, `device-width` is busted. Don't use it. Use `width` instead.
