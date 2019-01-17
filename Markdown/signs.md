---
html:
  embed_local_images: true
  embed_svg: true
  offline: false
  toc: false

print_background: false
---
<!--top mark-->

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [需要转义的字符](#需要转义的字符)
* [希腊字母](#希腊字母)
* [数学符号](#数学符号)
	* [其他符号](#其他符号)
	* [四则运算](#四则运算)
	* [（正）上下标](#正上下标)
	* [箭头](#箭头)
	* [连线/hat](#连线hat)
	* [逻辑运算](#逻辑运算)
	* [微积分运算](#微积分运算)
	* [三角运算](#三角运算)
	* [指数对数](#指数对数)
	* [集合运算](#集合运算)
	* [公式大括号](#公式大括号)

<!-- /code_chunk_output -->

# 需要转义的字符
|符号|$\#$|$\%$|$\&$|$\_$|$\{$|$\}$|$\$$|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|语法|\\#|\\%|\\&|\\_|\\{|\\}|\\$|


# 希腊字母

|符号|$\alpha$|$\beta$|$\gamma$|$\Gamma$|$\delta$|$\Delta$|$\epsilon$|$\varepsilon$|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|语法|\alpha|\beta|\gamma|\Gamma|\delta|\Delta|\epsilon|\varepsilon|
|符号|$\zeta$|$\eta$|$\theta$|$\Theta$|$\vartheta$|$\iota$|$\kappa$|$\lambda$|
|语法|\zeta|\eta|\theta|\Theta|\vartheta|\iota|\kappa|\lambda|
|符号|$\Lambda$|$\mu$|$\nu$|$\xi$|$\Xi$|$\pi$|$\Pi$|$\varpi $|
|语法|\Lambda|\mu|\nu|\xi|\Xi|\pi|\Pi|\varpi |
|符号|$\rho$|$\varrho$|$\sigma$|$\Sigma$|$\varsigma$|$\tau$|$\upsilon$|$\Upsilon$|
|语法|\rho|\varrho|\sigma|\Sigma|\varsigma|\tau|\upsilon|\Upsilon|
|符号|$\phi$|$\Phi$|$\varphi$|$\chi$|$\psi$|$\Psi$|$\Omega$|$\omega$|
|语法|\phi|\Phi|\varphi|\chi|\psi|\Psi|\Omega|\omega|

[返回目录](#top)

# 数学符号

<!---[返回目录](#top)--->

## 其他符号

|符号|$\cdot$|$\dots$|$\ast$|$\circ$|$\bigodot$|$\bigotimes$|$\leq$|$\geq$|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|语法|\cdot|\dots|\ast|\circ|\bigodot|\bigotimes|\leq|\geq|
|符号|$\neq$|$\approx$|$\equiv$|$\sum$|$\prod$|$\coprod$|$\prime$|$\vec{a}$|
|语法|\neq|\approx|\equiv|\sum|\prod|\coprod|\prime|\vec{a}|
|符号|$\sqrt[n]{a}$|$\sqrt{a}$|$\ldots$|$\dots$|$\cdots$|$\frac{a}{b}$|${{a}\over{b}}$|$\neq$|
|语法|\sqrt[n]{a}|\sqrt{a}|\ldots|\dots|\cdots|\frac{a}{b}|{{a}\over{b}}|\neq|

`\cdots`,`\cdot`用于公式输入，`\dots`,`\ldots`用于正文内容插入
[返回目录](#top)

## 四则运算


|符号|$\times$|$\div$|$\pm$|$\mid$|$+$|$-$|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|语法|\times|\div|\pm|\mid|+|-|

[返回目录](#top)

## （正）上下标
|符号|$a^{bcd}s$|$a^bc$|$a_{bcs}$|$\min\limits_{abc}$|$a\stackrel{c}{\rightarrow}b$|
|:-:|:-:|:-:|:-:|:-:|:-:|
|语法|a^{bcd}s|a^bc|a_{bcs}|\min\limits_{abc}|a\stackrel{c}{\rightarrow}b|

|符号|语法|
|:-:|:-:|
|$\underrightarrow{\text{text}}$|\underrightarrow{\text{text}}|
|$A \xleftarrow{n=0} B \xrightarrow[T]{n>0} C$|A \xleftarrow{n=0} B \xrightarrow[T]{n>0} C|
|$\triangleq$|\triangleq|
|$\underset{0\leq j \leq k-1}{\arg\min}$|\underset{0\leq j \leq k-1}{\arg\min}|
|$\underset{A}{top}$|\underset{A}{top}|
|$\overset{a}{bottom}$|\overset{a}{bottom}|
$\stackrel{a}{bot}$|\stackrel{a}{bot}|

`\limits`控制下标是在符号的右侧还是下侧，如$\sum^{n}_{1}a_i$，有`\limits`：$\sum\limits^{n}_{1}a_i$，支持的符号有：`\sum`,`\prod`,`\max`,`\min`
[返回目录](#top)

## 箭头

|符号|$\uparrow$|$\downarrow$|$\leftarrow$|$\rightarrow$|$\Uparrow$|$\Downarrow$|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|语法|\uparrow|\downarrow|\leftarrow|\rightarrow|\Uparrow|\Downarrow|
|符号|$\Leftarrow$|$\Rightarrow$|$\longleftarrow$|$\longrightarrow$|$\Longleftarrow$|$\Longrightarrow$|
|语法|\Leftarrow|\Rightarrow|\longleftarrow|\longrightarrow|\Longleftarrow|\Longrightarrow|

[返回目录](#top)

## 连线/hat

|符号|$\overline{a+b+c+d}$|$\underline{a+b+c+d}$|$\hat{y}$|$\check{y}$|$\breve{y}$|
|:-:|:-:|:-:|:-:|:-:|:-:|
|语法|\overline{a+b+c+d}|\underline{a+b+c+d}|\hat{y}|\check{y}|\breve{y}|

|符号|$\overbrace{a+\underbrace{b+c}_{1.0}+d}^{2.0}$|$\widehat{abc}$|$\widecheck{abc}$|
|:-:|:-:|:-:|:-:|
|语法|\overbrace{a+\underbrace{b+c}_{1.0}+d}^{2.0}|\widehat{abc}|\widecheck{abc}|
[返回目录](#top)

## 逻辑运算

|符号|$\because$|$\therefore$|$\forall$|$\exists$|
|:-:|:-:|:-:|:-:|:-:|:-:|
|语法|\because|\therefore|\forall|\exists|
[返回目录](#top)

## 微积分运算
|符号|$y{\prime}x$|$\int$|$\iint$|$\iiint$|$\oint$|$\lim$|$\infty$|$\nabla$|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|语法|y{\prime}x|\int|\iint|\iiint|\oint|\lim|\infty|\nabla|

**$\lim_{n\rightarrow+\infty}\frac{1}{n(n+1)}$**：`\lim_{n\rightarrow+\infty}\frac{1}{n(n+1)}`
[返回目录](#top)

## 三角运算

|符号|$\bot$|$\angle$|$30^\circ$|$\sin$|$\cos$|$\tan$|$\cot$|$\sec$|$\csc$|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|语法|\bot|\angle|30^\circ|\sin|\cos|\tan|\cot|\sec|\csc|
[返回目录](#top)

## 指数对数

|符号|$\log$|$\lg$|$\ln$|$\log_24$|
|:-:|:-:|:-:|:-:|:-:|
|语法|\log|\lg|\ln|\log_24|

[返回目录](#top)

## 集合运算

|符号|$\emptyset$|$\in$|$\notin$|$\subset$|$\supset$|$\subseteq$|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|语法|\emptyset|\in|\notin|\subset|\supset|\subseteq|
|符号|$\supseteq$|$\bigcap$|$\bigcup$|$\bigvee$|$\bigwedge$|$\biguplus$|
|语法|\supseteq|\bigcap|\bigcup|\bigvee|\bigwedge|\biguplus|

[返回目录](#top)

## 公式大括号

对于公式 $f(x)=\begin{cases}
    0& x=0\\
    1& x\neq 0
\end{cases}$ ，可以有以下三种写法：

```markdown 
$f(x)=\begin{cases}
    0& x=0\\
    1& x\neq 0
\end{cases}$
```

```markdown
$f(x)=\left\{\begin{array}{ll}
    0& x=0\\
    1& x\neq 0
\end{array}\right.$
```

```markdown
$f(x)=\left\{\begin{aligned}
    0& x=0\\
    1& x\neq 0
\end{aligned}\right.$
```

>使用`&`对齐
<!---
|符号|$$|$$|$$|$$|$$|$$|$$|$$|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|语法|||||||||
|符号|$$|$$|$$|$$|$$|$$|$$|$$|
|语法|||||||||
--->
