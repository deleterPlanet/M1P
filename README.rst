|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Оптимизация количества деревьев в ансамбле градиентного бустинга с использованием стратегий выбора агрегирующих функций
    :Тип научной работы: M1P
    :Автор: Якушевич Антон Сергеевич
    :Научный руководитель: д.ф-м.н., в. науч. сотр. ВЦ РАН Сенько Олег Валентинович
    :Научный консультант: -

Abstract
========

В работе исследуется задача оптимизации ансамблей градиентного бустинга за счёт выбора обобщённой агрегирующей функции. Исследование проводится для уменьшения количества деревьев в ансамбле с целью повышения интерпретируемости модели при сохранении её прогностической точности. Для этого предлагается использовать усечённое разложение агрегирующей функции в ряд Тейлора и оптимизировать её коэффициенты с помощью градиентного спуска, что позволяет адаптивно настраивать структуру ансамбля и улучшать баланс между точностью и интерпретируемостью. Экспериментальные результаты показывают, что оптимизация коэффициентов ряда Тейлора позволяет сократить число деревьев в ансамбле при сохранении точности модели, а также повышает устойчивость и эффективность обучения по сравнению с классическим градиентным бустингом.

Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. 

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.
