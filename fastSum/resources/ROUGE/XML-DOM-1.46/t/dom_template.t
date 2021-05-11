BEGIN {print "1..2\n";}
END {print "not ok 1\n" unless $loaded;}
use XML::DOM;
$loaded = 1;
print "ok 1\n";

my $test = 1;
sub assert_ok
{
    my $ok = shift;
    print "not " unless $ok;
    ++$test;
    print "ok $test\n";
    $ok;
}

#Test 2

print "ok 2\n";
