---
layout: post
author: initialxy
title: "Getting the Right Time Zone in Python"
description: "Time Zone can be tricky even in Python. Here is how to get the right one."
category: "Lesson"
tags: [Python, Time Zone, System, Debugging]
---
{% include JB/setup %}

Given the [recent news](https://www.nytimes.com/2022/03/15/us/politics/daylight-saving-time-senate.html) that most of the US will stick with daylight saving time starting 2023, I'd like to revisit how error-prone dealing with time zone could be in software. Even in Python, which is supposed to be a more newbie-friend and intuitive languages out there, it does a rather confusing job at at. I hope to write this post as more of a straight to point code pointers for how to deal with time zones in Python with a more humanly readable documentation instead of pages of cryptic API documentation. In my professional experience, I have delt with countless time zone and daylight savings related bugs literally every year. There's always something that can go wrong. Being oncall during daylight savings change is always a time. 

Now let me get straight to point by looking at some sample code for common use cases. I will elaborate in a later section. For all of these examples, keep in mind that I live in America/Los_Angeles, which I will use as reference point.

### 1. Format date and time given a time zone
Suppose you are given a timestamp or `datetime`, and you want to show user living in another time zone what that time is in their time zone. A common use case to consider is flight tickets, where departure time and arrival time are often shown in two different time zones. Here is the best way to do it

```python
from pytz import timezone
from datetime import datetime

def format(dt, tz):
  return dt.astimezone(tz).isoformat() # or strftime

tz = timezone("America/New_York")
dt = datetime.fromtimestamp(1647642342)

format(dt, tz) # 2022-03-18T18:25:42-04:00
```

### 2. Get timestamp for a date and time given a time zone
Basically the reverse of the above. Consider a use case, where you ask a user to enter their preferred reservation time in another time zone. Here is how you'd get the correct timesamp.

```python
from pytz import timezone
from datetime import datetime

def timestamp(iso_str, tz)
  dt = datetime.fromisoformat(iso_str) # or strptime
  return tz.localize(dt).timestamp()

tz = timezone("America/New_York")
timestamp("2022-03-01T09:00", tz) # 1646143200.0
```

### 3. Get timestamp of a time on a certain date and time zone
Suppose you know a store opens at 9:00 every day, and you want to find out when it will open on a particular date as a timestamp. This is closely related on the above. You can see that one way to do this is to simply concat an ISO date time string and use the method above. But here is a more elegant solution.

```python
from pytz import timezone
from datetime import datetime, date, time

def timestamp(d, time_str, tz)
  dt = datetime.combine(d, time.fromisoformat(time_str))
  return tz.localize(dt).timestamp()

tz = timezone("America/New_York")
d = date.fromisoformat("2022-03-01")
timestamp(d, "09:00", tz) # 1646143200.0
```

### 4. Format date and time in another timezone given a datetime
Suppose you have a `datetime` instance passed to you, and you want to display it in a different timezone while preserving its timestamp.

```python
from pytz import timezone
from datetime import datetime

def format(dt, tz):
  return dt.astimezone(tz_other).isoformat()

tz_og = timezone("America/Los_Angeles")
tz_other = timezone("America/New_York")
dt = datetime.now(tz_og)
format(dt, tz_other) # 2022-03-18T19:05:08.905647-04:00
```

### 5. Get timestamp of a datetime in another time zone while preseving local time
Now this is a tricky one and hopefully doesn't happen frequently in practice. Suppose you were given a datetime that represent the air time of an episode of a TV show, and you are asked to compute when it needs to air in a different time zone at the same local time (eg. 7 PM in Los Angeles and 7 PM in New York). You may be tempted to try a bunch of different thing, but here is the correct solution.

```python
def timestamp(dt, tz):
  return tz.localize(dt.replace(tzinfo=None)).timestamp()

tz_og = timezone("America/Los_Angeles")
tz_other = timezone("America/New_York")
dt = tz_og.localize(datetime.fromisoformat("2022-03-01T19:00"))
timestamp(dt, tz_other) # 1646179200.0
```

<div class="preview_img_3" markdown="1">

## Why Time Zone is So Hard
I think many of us wouldn't think too much about time zone. How hard can it possibly be? You get a timestamp and just offset a few hours here and there from UTC right? Well not so simple. The key issue here is that time zones can be changed at any time by a local goverment. For example, as I linked above, the US Senate decided to stick with daylight saving time starting 2023. Another amusing example is when [Turkey changed their daylight saving rule](https://support.microsoft.com/en-us/topic/turkey-ends-dst-observance-56f14484-a323-543c-5e36-f701723f5b22) and [Tesla apparently didn't update for at least two years](https://www.reddit.com/r/teslamotors/comments/ag6r2f/please_help_our_turkish_tesla_community_reach/). This is the reason why there's GMT and UTC. GMT is a real-life time zone, which is subject to change by the UK government, while UTC is a global reference point, and not a real time zone. For now GMT and UTC are equal, but that's not guranteed. Imagine you were using GMT-7, then the next day the UK government decides to forward it by an hour for some reason. All of your references would become incorrect. So an **UTC offset is only useful when referencing to an exact point in history**. If you were to change that reference time, then it could cause errors.

Of course the other reason why time zone is so difficult is because of daylight savings. Needless to say, your offset could be different depending on when and where you are referencing. Take a look at this CGP Grey video that explains some finer details about it. 

[![Daylight Saving Time Explained](https://img.youtube.com/vi/84aWtseb2-4/0.jpg)](https://www.youtube.com/watch?v=84aWtseb2-4)

Again, like I already mentioned above. All of these are subject to change at any time. So to correctly understand date and time, you not only need to know a timestamp, but also a **location** along with **its history**. Fortunately there's a database that's being maintained for this exact set of information, and that's the [tz database](https://en.wikipedia.org/wiki/Tz_database). Most programming languages would offer some way to easily import this information. In case of Python, that's supplied by [pytz](https://pypi.org/project/pytz/) or [dateutil](https://dateutil.readthedocs.io/en/stable/). In this post, I'd like to use `pytz` with Python's `datetime` module, because I feel that's more bare metal, and also unfortunately error-prone. In order to use the tz database, you will need to have a supported [time zone name](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones), eg. America/Los_Angeles. You might also wonder what's the difference between time zone and tz time zone name. Time zones, such as Pacific time zone could include regions across multiple jurisdictions. They might attempt to keep in sync, but that's not guaranteed. Take a look at [Pacific time's Wikipedia page](https://en.wikipedia.org/wiki/Pacific_Time_Zone) for instance.

> Effective in the U.S. in 2007 as a result of the Energy Policy Act of 2005, the local time changes from PST to PDT at 02:00 LST to 03:00 LDT on the second Sunday in March and the time returns at 02:00 LDT to 01:00 LST on the first Sunday in November. The Canadian provinces and territories that use daylight time each adopted these dates between October 2005 and February 2007. In Mexico, beginning in 2010, the portion of the country in this time zone uses the extended dates, as do some other parts.

Therefore to correctly understand date and time, you will need a much more granular reference. Hence the tz time zone name is used.

## Python's Many Quirks and Pitfalls
I'd like to point out why Python makes it so error-prone. Let's take a look at this paragraph from [pytz's documentation.](https://pypi.org/project/pytz/).

> This library only supports two ways of building a localized time. The first is to use the `localize()` method provided by the pytz library. This is used to localize a naive datetime (datetime with no timezone information): ... The second way of building a localized time is by converting an existing localized time using the standard `astimezone()` method: ... Unfortunately using the tzinfo argument of the standard datetime constructors ‘’does not work’’ with pytz for many timezones.

When you create a `timezone` instance with `pytz`. it does not know what offset it should use yet. Remember, there's a history to each time zone. Its offset can only be determined when it can reference to a point in history. So by default pytz uses [local mean time](https://en.wikipedia.org/wiki/Local_mean_time), which is entirely useless in most applications. Take a look here.

```python
from pytz import timezone
timezone("America/Los_Angeles")
<DstTzInfo 'America/Los_Angeles' LMT-1 day, 16:07:00 STD>
```

If you were to use this instance directly without referencing to a data and time then you will get wildly strange results. This is what it meant by "does not work" in its documentation. For example.

```python
datetime(2022, 1, 1, 0, 0, tzinfo=timezone("America/Los_Angeles")).isoformat()
'2022-01-01T00:00:00-07:53'
```

Notice its offset, that doesn't look right does it? Turns out `pytz` does not actually get the correct offset even when put into a `datetime` instance, and there's no warning about it. So the correct way to do this is to use its `timezone.localize()` in this case.

```python
timezone("America/Los_Angeles").localize(datetime(2022, 1, 1, 0, 0)).isoformat()
'2022-01-01T00:00:00-08:00'
```

That looks much better. There is actually a third way that will force `pytz` to get a correct offset, which is `datetime.now(tz)`. If you look at the `tzinfo` property of a `datetime` instance that's glued with a time zone (so called aware `datetime`), you will notice that it has the correct offset.

```python
tz = timezone("America/Los_Angeles")
now = datetime.now(tz)
tz
<DstTzInfo 'America/Los_Angeles' LMT-1 day, 16:07:00 STD>
now.tzinfo
<DstTzInfo 'America/Los_Angeles' PDT-1 day, 17:00:00 DST>
```

You may be tempted to take a corrected `timezone` instance and put it into another `datetime`. Don't do it! Remember, offset is only useful when referencing to an exact time in history. If you simply copy it over, it will produce the wrong result. For example, at America/Los_Angeles, daylight savings started on March 5, 2022. 

```python
tz = timezone("America/Los_Angeles")
now = datetime.now(tz)
datetime(2022, 1, 1, tzinfo=dt.tzinfo).isoformat()
'2022-01-01T00:00:00-07:00'
```

Notice that this is not correct. The correct result should have been 2022-01-01T00:00:00-08:00, because on January 1st, America/Los_Angeles should be using standard time.

So what's the difference between `localize()` and `astimezone()`. If you take a look at [datetime's documentation](https://docs.python.org/3/library/datetime.html), it talks about naive vs aware datetime. A `datetime` instance can be created without associating it with a `timezone`, in which case it's considered naive. When it is associated with a `timezone`, then it's aware. Keep in mind that Python's built-in `datetime.timezone` does not contain tz database, so it's not particularly useful in most applications. However even with a naive `datetime` instance you could still get a timestamp out of it. That's because even without time zone, `datetime` implicitly references your local offset. When you need to create a naive `datetime` instanace to be aware, use `localize()`. In this case date and time are preserved instead of timestamp. When you need to change time zone of an already aware `datetime` instance, use `astimezone()`. Timestamp is preserved instead of date and time. If you use `localize()` on an aware `datetime` instance, it will throw an exception.

```python
dt = datetime(2022, 1, 1)
dt.timestamp() # I'm in America/Los_Angeles
1641024000.0
dt.isoformat()
'2022-01-01T00:00:00'

tz = timezone("America/New_York")
dt = tz.localize(dt)
dt.timestamp()
1641013200.0
dt.isoformat()
'2022-01-01T00:00:00-05:00'

dt = dt.replace(tzinfo=None) # Converts aware to naive
dt.timestamp()
1641024000.0
dt.isoformat()
'2022-01-01T00:00:00'

dt = dt.astimezone(tz)
dt.timestamp()
1641024000.0
dt.isoformat()
'2022-01-01T03:00:00-05:00'

dt = dt.astimezone(timezone("America/Los_Angeles"))
dt.timestamp()
1641024000.0
dt.isoformat()
'2022-01-01T00:00:00-08:00'

tz.localize(dt)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/initialxy/git/initialxy-dashboard/venv/lib/python3.10/site-packages/pytz/tzinfo.py", line 318, in localize
    raise ValueError('Not naive datetime (tzinfo is already set)')
ValueError: Not naive datetime (tzinfo is already set)

now = datetime.now() # why you should use datetime.now(tz)
now.timestamp()
1647672125.937841
now.isoformat()
'2022-03-18T23:42:05.937841'
wrong_now = tz.localize(now)
wrong_now.timestamp()
1647661325.937841
wrong_now.isoformat()
'2022-03-18T23:42:05.937841-04:00'

now.astimezone(tz).timestamp()
1647672125.937841
```

## Conclusion
So to summary the above into rules of thumb:
* In order to correctly understand date and time, you will need two pieces of information: a refrence in history such as timestamp and a tz database time zone name. This is universal in any programming language. Even if your location does not observe daylight savings, you may stil have to reference to a point in the past where offset could be different than what it is now.
* Just avoid `tzinfo` field in Python's `datetime` API at all cost. It will only create confusions.
* When you need to preserve date and time, and create an aware `datetime` instance, use `localize()`. Just make sure you start with a naive `datetime` instance. Use `replace(tzinfo=None)` to convert it back to naive if needed.
* When you need to preserve timstamp and set or change time zone, use `astimezone()`.