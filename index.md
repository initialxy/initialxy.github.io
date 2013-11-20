---
layout: page
title: Brain, what do you want to do tonight?
---
{% include JB/setup %}

<p class="lead">
    The same thing we do every night, dude&ndash;try to squash stupid bugs!
</p>

<div class="posts_summary">
  {% for post in site.posts %}
    <div class="post_summary">
        <h3>
            <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a>
        </h3>
        <small class="info">
            {{ site.authors[post.author].display_name }},
            {{ post.date | date: "%B %-d %Y" }}
        </small>

        <p class="excerpt">{{ post.excerpt | markdownify }}</p>
        <a href="{{ BASE_PATH }}{{ post.url }}">Read More</a>
    </div>
  {% endfor %}
</div>

