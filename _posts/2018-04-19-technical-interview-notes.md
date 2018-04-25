---
layout: post
author: initialxy
title: "Technical Interview Notes"
description: "Technical interveiw review notes"
category: "notes"
tags: [Programming]
---
{% include JB/setup %}

It's that time again. Now you gotta prepare for technical interviews, which you haven't done in years. Here are some review notes that should come in handy to remind yourself of the thrill. <span class="hidden">read more</span>

### Fundamentals
Straight off Wikipedia. It would be a shame if you failed an interview just because you forgot some common knowledge.
* [Software design patterns](https://en.wikipedia.org/wiki/Software_design_pattern)
* [Tree](https://en.wikipedia.org/wiki/Tree_(graph_theory))
* [Binary tree](https://en.wikipedia.org/wiki/Binary_tree)
* [Binary search tree](https://en.wikipedia.org/wiki/Binary_search_tree)
* [AVL tree](https://en.wikipedia.org/wiki/AVL_tree) A concret implementation of a self-balancing binary search tree.
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
* [Careercup](https://www.careercup.com/page). I'd short my most recent, as well as votes and try to go through as many as possible. Keep in mind that tech companies often retire questions that's easily found on the internet. So there is a good chance that highest voted questions won't be asked, but it's still a good idea to absorb their spirits.
* [Glassdoor](https://www.glassdoor.com/Interview/index.htm)
* [Cracking the Coding Interview: 150 Programming Questions and Solutions](https://www.amazon.com/Cracking-Coding-Interview-Programming-Questions/dp/098478280X) Again, keep in mind that the exact questions probably won't be asked, as this is an extremely popular book (I somehow ended up with two physical copies). Just absorb its spirit.

### Practices
Different tech companies run their interviews differently, but there are some common patterns. Prepare yourself for these patterns and practice ahead of time.
* Build up your stemina. Typically you should expect multiple rounds of back to back interviews (usually 5, though I've seen 6). This can be extremely exhausting and will require some stemina and mental preparation. Sleep well, eat well, and run mock interviews with your buddies at least once to grind yourself until exhaustion. (Remember to treat your friend a nice meal, because you are wasting a whole day of your friend's time.)
* Work your muscles. During an interview, you will be expected to write on a board or wall (depending on how cool the meeting room is). You actually **need** some arm muscles and stemina. Prepare your muscles during a mock interview by actually writing on a board. Remember to write reasonably small and leave blank lines between each time and some blank spaces between variables, so you can easily edit it as you think. You don't have a text editor, editing a line could be painful.
* Prepare and practice the usual questions. Depending on the tech company you are interviewing for, these questions can sometimes be asked for different reasons. Some companies take these questions into their interview evaluation, while others use these only as warm up. Some companies have dedicated interview sessions for these personal questions, sometimes with HR. It's a good idea to ask your recruiter and clarify their significance ahead of time if possible. Prepare a short answer and long answer. Say it out loud and get used to answering them. In case these questions don't hold much (if any) significance, don't waste time on these. Just be concise and short.
    * Tell me about yourself.
    * What do you feel passionate about?
    * Wha't your proudest project and why?
    * How do you demonstrate your leadership skills?
    * What are you looking for at the company you are interview for?
    * Why do you want to leave your current post?
* Different tech companies run their interviews differently, but usually you'd start with some personal questions (see above point), then proceeds to coding questions. Keep in mind that interviewers will probably want to ask **multiple** coding/algorithm questions. You will definitely be asked to code for your first solution, though you may or may not be asked to code the follow up questions. So there are a few things to remember:
    * Don't over-think or waste time on the first question. It's not supposed to be that hard. Deliver a short and sweet answer that's easy to code. As long as it's faster than O(n^2), it's usually good enough. If it's not good enough, then interviewer will let you know.
    * Once you are done the first question, the interviewer will want to move on to a follow up question, which may be built on top of the first question or may be a completely different question. Expect this one to be harder, but you may or may not be asked to code it. Work out the solution first without coding. Use visual aides to help you explain. Only code if interviewer asks you to code. When solving the question, feel free to discuss with the interviewer and expect to have back and forth dialogs with the interviewer. The interviewer is supposed to guide you along the right path.
    * To reinstate an above point, leave plenty of blank spaces when you are writing on board. In case the follow up question is built on top of the first question, you want to be able to easily make edits on a board.
* Prepare to answer "trick" questions. There is a good chance that you will be asked a question that has no perfect answer or no practical perfect answer. For example NP problems or database problems that (implicitly) demands everything (consistency, low latency, redundency, availability). You must be able to quickly identify that you got an "impractical" question. For example, recognized that the question is a variation of the [Knapsack question](https://en.wikipedia.org/wiki/Knapsack_problem). You are supposed to tackle these types of questions by first negotiating with the interviewer to determine what requirements to prioritize. At the end, interviewer may tell you to choose your priorities, in which case you must be able to justify your decision. There may not be a good or bad answer, just how you justify it. Then you may need to deliver a best effort approximation of the solution. Use techniques like [sampling](https://en.wikipedia.org/wiki/Sampling_(statistics)), [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing), [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) or whatever your smarts get you. Briefly discuss the compromises you are making and why they are acceptable. For example "There's a 1% probability that the worst case may happen. However if it does happen, the system as a whole won't crash, though there may be an increased latency. Since we have estabilished that this is not a mission critical real time system, 1% unusually long latency is acceptable."
* Prepare to answer open ended design questions. Again, these questions expect you negotiate with the interviewer, and there's often no perfect solution. There's no good or bad answer just how you justify your decisions. However keep in mind how you deliver your justifications. Simply saying "Yeah because that's totally important" is not a justification. You want to discuss how you can gather evidence to support your decision. You probably don't have the actual evidences, but you need to be able to state your estimates and methodology. Think the followings:
    * Make estimates. Find your baseline (if possible) and go from there. eg. We will take last 5 years worth of Back Friday sales traffic pattern as baseline. We observe that the p99 user purchased 15 items in one cart last year. We can estimate that this year we can expact a 10% increase. Therefore our design must be able to handle the estimated traffic pattern. Suppose we allow 0.1% timeout rate, in which case client software is expected to retry ...
    * State your methodology to collect metrics in order to prove that your solution meets requirements. eg. The biggest bottleneck is likely the RAM on each node. We estimate each node will need to be able to handle 10k queries per second. We can deploy a prototype on a VM with similar configuration, sample a portion of the logs from last year's traffic and stress test on our VM to see if RAM is in fact the bottleneck and at which point it will break.
* When are you asked a question that you don't have an immediate answer, don't panic. Try to break down the question into sub-problems. Ask yourself what smaller questions you can ask to help. Usually there's one sub-problem that's tricky. Try to find a solution to that. This is why you should read sample interview question even if the exact question will not be asked. When I said "absorb its spirit", I really meant understand as many solutions to as many sub-problems as possible. You want to build up your arsenal of sub-problems.
* You may be asked a question that's seamingly easy to solve but cumbersome to code. eg. lots of edge cases, lots of small utilities that need to be applied all over the place. These questions are designed to test your coding skill rather than problem solving skill. You want to write your solution as **elegantly** as possible. Take a deep breathe and think through on the layout of your code but not necessarily the exact code. Plan it well and again, leave plenty of empty space for edits. Don't start coding without thinking how it's gonna finish. Use helper functions as much as it is elegant. Use language features to improve readability (if you are writing Python, then you got plenty).