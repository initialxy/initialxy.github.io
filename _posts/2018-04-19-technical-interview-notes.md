---
layout: post
author: initialxy
title: "Technical Interview Notes"
description: "Technical interveiw review notes"
category: "Notes"
tags: [Programming, Interview]
---
{% include JB/setup %}

It's that time again. Now you gotta prepare for technical interviews, which you haven't done in years. Here are some review notes that should come in handy to remind yourself of the thrill. This note is meant for my own future reference. But since it contains no confidential materials, I decided to publish it on my blog.<span class="hidden">read more</span>

### Fundamentals
Straight off Wikipedia. It would be a shame if you failed an interview just because you forgot some common knowledge.
* [Software design patterns](https://en.wikipedia.org/wiki/Software_design_pattern)
* [Tree](https://en.wikipedia.org/wiki/Tree_(graph_theory))
* [Binary tree](https://en.wikipedia.org/wiki/Binary_tree)
* [Binary search tree](https://en.wikipedia.org/wiki/Binary_search_tree)
* [AVL tree](https://en.wikipedia.org/wiki/AVL_tree) A concreteimplementation of a self-balancing binary search tree.
* [Trie](https://en.wikipedia.org/wiki/Trie)
* [Graph theory](https://en.wikipedia.org/wiki/Graph_theory)
* [Shortest path problem](https://en.wikipedia.org/wiki/Shortest_path_problem)
* [Knuth-Morris-Pratt algorithm](https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm)
* [Dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) A lot of people, even veterans, seem to struggle with understanding dynamic programming. Here's how I'd explain it in one sentence: [memoize](https://en.wikipedia.org/wiki/Memoization) deterministic functions called repeatedly with the same parameters.
* [Async function](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function)
* [P versus NP problem](https://en.wikipedia.org/wiki/P_versus_NP_problem)

I'd prefer to interview in Java, which is a very well universally understood language with least amount of magic. Here are some review notes for Java utilities in case you haven't been working on Java for a while.
* [Java 8 cheat sheet](https://zeroturnaround.com/rebellabs/java-8-best-practices-cheat-sheet/)
* [Scanner](https://docs.oracle.com/javase/7/docs/api/java/util/Scanner.html)
* [Pattern](https://docs.oracle.com/javase/7/docs/api/java/util/regex/Pattern.html) Review some regular expressions while at it.
* [Java Syncronization](https://docs.oracle.com/javase/tutorial/essential/concurrency/sync.html)

### Interview questions
* If you currently work for a company that has its own internal wiki of interview questions, then start there.
* [Careercup](https://www.careercup.com/page). I'd sort by most recent, as well as votes thentry to go through as many questions as possible. Keep in mind that tech companies often retire questions that's easily found on the internet. So there is a good chance that highest voted questions won't be asked, but it's still a good idea to absorb their spirits.
* [Glassdoor](https://www.glassdoor.com/Interview/index.htm)
* [Cracking the Coding Interview: 150 Programming Questions and Solutions](https://www.amazon.com/Cracking-Coding-Interview-Programming-Questions/dp/098478280X) Again, keep in mind that the exact questions probably won't be asked, as this is an extremely popular book (I somehow ended up with two physical copies). Just absorb its spirit.

### Tips and tricks
Different tech companies run their interviews differently, but there are some common patterns. Prepare yourself for these patterns and practice ahead of time. These are the tips and tricks that I can think of based on my experience as an interviewee as well as an interviewer. This is not meant to be a comprehensive guide. These are just the things off the top of my head.

#### Build up your stamina
Typically you should expect multiple rounds of back to back interviews (usually 5, though I've seen 6). This can be extremely exhausting and will require some stamina and mental preparation. Sleep well, eat well, and run mock interviews with your friends at least once to grind yourself until exhaustion. (Remember to treat your friend a nice meal, because you are wasting a whole day of their time.)

#### Work your muscles
During an interview, you will be expected to write on a board or wall (depending on how cool the meeting room is). You actually **need** some arm muscles and stamina. Prepare your muscles during a mock interview by actually writing on a board. Remember to write reasonably small and leave blank lines between each line and some blank spaces between variables, so you can easily edit it as you think. You won’t have a text editor, so editing a line could be painful.

#### Prepare and practice the usual questions
Depending on the tech company you are interviewing for, questions regarding your expertise can be asked for different reasons. Some companies take these questions into their interview evaluation, while others use these only as warm up. Some companies have dedicated interview sessions for these. It's a good idea to ask your recruiter and clarify their significance ahead of time if possible. Prepare a short answer and a long answer. Say it out loud and get used to answering them. In case these questions don't hold much (if any) significance, don't waste time on these. Just be concise and short.
* Tell me about yourself.
* What do you feel passionate about?
* What’s your proudest project and why?
* How do you demonstrate your leadership skills?
* What are you looking for at the company you are interview for?
* Why do you want to leave your current post?

#### Use time wisely
Different tech companies run their interviews differently, but usually you'd start with some personal questions (see above point), then proceed to coding questions. Keep in mind that interviewers will probably want to ask **multiple** coding/algorithm questions. If you spend the entire session on just the first question, they you have likely failed on the spot. You will definitely be asked to code for your first solution, though you may or may not be asked to code the follow up questions. So there are a few things to remember:
* Don't overthink or waste time on the first question. It's not supposed to be that hard. Deliver a short and sweet answer that's easy to code. As long as it's faster than O(n^2), it's usually good enough. If it's not good enough, then the interviewer will let you know.
* Once you are done the first question, the interviewer will want to move on to a follow up question, which may be built on top of the first question or may be a completely different question. Expect this one to be harder, so save plenty of time for it. But you may or may not be asked to code it. Work out the solution first without coding. Use visual aides to help you think and explain. Only code if the interviewer asks you to code. When solving the question, feel free to discuss with the interviewer and expect to have back and forth dialogs. The interviewer is supposed to guide you along the right path.
* To reinstate an above point, leave plenty of blank spaces when you are writing on board. In case the follow up question is built on top of the first question, you want to be able to easily make edits on a board.

#### Prepare to answer "trick" questions
There is a good chance that you will be asked a question that has no perfect answer or no practical perfect answer. For example NP problems or database problems that (implicitly) demands everything (consistency, low latency, redundancy, availability). You must be able to quickly identify that you are asked an "impractical" question. For example, recognized that the question is a variation of the [Knapsack question](https://en.wikipedia.org/wiki/Knapsack_problem). You are supposed to tackle these types of questions by first negotiating with the interviewer to determine what requirements to prioritize. At the end, interviewer may tell you to choose your priorities, in which case you must be able to justify your decision. There may not be a good or bad answer, just how you justify it. Then you may need to deliver a best effort approximation of the solution. Use techniques like [sampling](https://en.wikipedia.org/wiki/Sampling_(statistics)), [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing), [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) or whatever your smarts get you. Briefly discuss the compromises you are making and why they are acceptable. eg. There's a 1% probability that the worst case may happen. However if it does happen, the system as a whole won't crash, though there may be an increased latency. Since we have established that this is not a mission critical real time system, 1% unusually long latency is acceptable.

#### Prepare to answer open ended design questions
Again, these questions expect you to negotiate with the interviewer, and there's often no perfect solution. There's no good or bad answer, just how you justify your decisions. However keep in mind how you deliver your justifications. Simply saying "Yeah because that's totally important" is not a justification. You want to discuss how you can gather evidence to support your decision. You probably don't have the actual evidences, but you need to be able to state your estimates and methodology. Think the followings:
* Make estimates. Find your baseline (if possible) and go from there. eg. We will take last 5 years worth of Back Friday Sale traffic pattern as baseline. We observe that the p99 user purchased 15 items in one cart last year. Based on past trends, we can estimate a 10% increase of traffic volume. Therefore our design must be able to handle the estimated traffic pattern. Suppose we tolerate 0.1% timeout rate, in which case client software is expected to retry ...
* State your methodology to collect metrics in order to prove that your solution meets requirements. eg. The biggest bottleneck is likely the RAM on each node. We estimate each node will need to be able to handle 10k QPS (queries per second). We can deploy a prototype on a VM with similar configurations. Sample a portion of the logs from last year's traffic and stress test on our VM to see if RAM is in fact the bottleneck and the maximum QPS it can handle before failing.

#### Prepare answers to sub-problems
When are you asked a question that you don't have an immediate answer, don't panic. Try to break it down sub-problems. Ask yourself what smaller questions you can ask to help solve the problem. Usually you will find one sub-problem that's tricky, for which you want to find a solution. This is why you should read sample interview questions even if the exact questions will not be asked. When I said "absorb its spirit", I really meant understand as many solutions to as many sub-problems as possible. You want to build up your arsenal of sub-problems.

#### Find hints in problem statements and observations
Before you begin, pay close attention to the problem statements, constraints and assumptions. These will often give away some hints. Be sure to ask the interviewer about any assumptions that you want to make, don't just quietly move forward without double checking implicit assumptions. Be sure not to solve for things that the problem didn't ask for. For example, if the problem asks for **any** one item that satisfies a constraint, then don't try to solve for all. Take advantage of properties of inputs. For example, if inputs are sorted, and you are looking for some items, then binrary search is likely the answer. If you are really stuck, take a deep breathe, step back and simply observe the problem and its sample outputs. Use visual aides and just write down many iterations of sample outputs or program states. Try to find patterns in them and use them for your advantage. For example, if you see outputs or program states that start to show some sort of repetition, then it can help you limit the number of iterations.

#### Write elegant code
You may be asked a question that's seemingly easy to solve but cumbersome to code. eg. Lots of edge cases. Lots of small utilities that need to be applied all over the place. These questions are designed to test your coding skill rather than problem solving skill. You want to write your solution as **elegantly** as possible. Take a deep breathe and think through the layout of your code but not necessarily the exact code. Plan it well and leave plenty of empty space for edits. Don't start coding without thinking how it's gonna finish. Use helper functions as much as it is elegant. Use language features to improve readability. (If you are writing Python, then you got plenty.)

#### Don’t worry about implementations of common utilities
If you want to use a well known utility or library, just go for it. You shouldn’t be asked to write its implementation on board. Any sane interviewer is not interested in wasting time knowing you can copy and paste some code from Wikipedia onto a board. However you must be able to explain what it does, its properties, its runtimes and why you choose to use it. Even if you don’t remember the exact library or API, you may ask the interviewer if it’s ok to assume an API exists. As long as it’s reasonably defined, interviewers shouldn’t be so picky about it. If they are picky, then it’s perfectly acceptable to ask the interviewer to define the API for you. Just don’t define a library that literally solves the question itself.