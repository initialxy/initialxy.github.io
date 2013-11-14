---
layout: page
title: Brain, what do you want to do tonight?
---
{% include JB/setup %}

<p class="lead">
    The same thing we do every night, Bro&ndash;try to squash stupid bugs!
</p>

<div class="posts">
  {% for post in site.posts %}
    <div class="post">
        <h3>
            <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a>
            <small class="date">{{ post.date | date_to_string }}</small>
        </h3>

        <p class="excerpt">{{ post.excerpt }}</p>
        <a href="{{ BASE_PATH }}{{ post.url }}">Read More</a>
    </div>
  {% endfor %}
</div>

