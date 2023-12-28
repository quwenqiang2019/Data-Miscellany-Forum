import itertools



data = [(['kibana', 'OCK', 'perl-HTTP-Body', 'puzzle-jigsaw', 'movim'], [None]),
        (['ngraph-gtk', 'ruby-tty-spinner', 'apertium-eu-es', 'ruby-sprockets', 'duma'], [None]),
        (['octave-linear-algebra', 'codequery', 'ruby-plist', 'dnss', 'crowdsec'], [None]),
        (['clickhouse', 'python-asdf', 'ruby-mixlib-config', 'ruby-fog-aws', 'ayatana-webmail'], [None]),
        (['lunar-date', 'dynamips', 'singularity-container', 'tuxtype', 'ruby-grit'], [None]),
        ([], [12])]

l = [[None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None]]
l = list(itertools.chain(*l))
print(l)

error_page_list = []
error_page_list.append('')
print(error_page_list)


my_list = [1, 2, 3]
my_list.extend([4, 5, 6])
print(my_list)
my_list.append([4, 5, 6])
print(my_list)