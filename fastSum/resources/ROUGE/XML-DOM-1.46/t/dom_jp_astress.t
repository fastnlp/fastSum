BEGIN {print "1..4\n";}
END {print "not ok 1\n" unless $loaded;}
use XML::DOM;
use CmpDOM;
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

sub filename
{
    my $name = shift;

    if ((defined $^O and
	 $^O =~ /MSWin32/i ||
	 $^O =~ /Windows_95/i ||
	 $^O =~ /Windows_NT/i) ||
	(defined $ENV{OS} and
	 $ENV{OS} =~ /MSWin32/i ||
	 $ENV{OS} =~ /Windows_95/i ||
	 $ENV{OS} =~ /Windows_NT/i))
    {
	$name =~ s!/!\\!g;
    }
    elsif  ((defined $^O and $^O =~ /MacOS/i) ||
	    (defined $ENV{OS} and $ENV{OS} =~ /MacOS/i))
    {
	$name =~ s!/!:!g;
	$name = ":$name";
    }
    $name;
}

# Test 2

my $parser = new XML::DOM::Parser;
unless (assert_ok ($parser))
{
    exit;
}

my $doc;
eval {
    $doc = $parser->parsefile (filename ('samples/minutes.xml'));
};
assert_ok (not $@);

my $doc2 = $doc->cloneNode (1);
my $cmp = new CmpDOM;
unless (assert_ok ($doc->equals ($doc2, $cmp)))
{
    print $cmp->context . "\n";
}
