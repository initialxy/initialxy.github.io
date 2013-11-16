---
layout: page
title: Brain, what do you want to do tonight?
---
{% include JB/setup %}

<p class="lead">
    The same thing we do every night, dude&ndash;try to squash stupid bugs!
</p>

<div class="posts_list">
  {% for post in site.posts %}
    <div class="post">
        <div class="row">
            <h3 class="col-md-10">
                <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a>
            </h3>
            <small class="date col-md-2">{{ post.date | date_to_string }}</small>
        </div>

        <p class="excerpt">{{ post.excerpt }}</p>
        <a href="{{ BASE_PATH }}{{ post.url }}">Read More</a>
    </div>
  {% endfor %}
</div>

